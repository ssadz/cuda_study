#include <iostream>
#include <cuda_runtime.h>

// 教学：Execution Configuration Optimizations
// - 使用 CUDA Occupancy API 计算活跃线程块（active blocks）和占用率（occupancy）
// - 探讨不同 block sizes 对核函数性能的影响

// 简单核函数：计算平方
__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * data[idx];
}

int main() {
    // 查询设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "  Warp size: " << prop.warpSize
              << ", Max threads per SM: " << prop.maxThreadsPerMultiProcessor
              << ", #SMs: " << prop.multiProcessorCount << std::endl;

    // 数据规模与分配
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int bs : blockSizes) {
        int activeBlocks = 0;
        // 计算每个 SM 上可运行的最大活跃 blocks
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, squareKernel, bs, 0);
        float occupancy = (activeBlocks * bs) /
            float(prop.maxThreadsPerMultiProcessor);

        // 启动配置
        int gridSize = activeBlocks * prop.multiProcessorCount;
        if (gridSize == 0) gridSize = (N + bs - 1) / bs;

        // 用 CUDA 事件测时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        squareKernel<<<gridSize, bs>>>(d_data, N);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "BlockSize=" << bs
                  << " ActiveBlocks/SM=" << activeBlocks
                  << " Occupancy=" << occupancy * 100 << "%"
                  << " Time=" << ms << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_data);
    return 0;
} 