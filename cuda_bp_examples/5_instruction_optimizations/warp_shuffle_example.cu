#include <iostream>
#include <cuda_runtime.h>

// Warp-level Shuffle 示例
// 教学：使用 __shfl_down_sync 在 warp 内部进行归约操作，避免使用共享内存
//      warpSize 通常为 32，shuffle 操作仅在同一 warp 的线程间有效

__global__ void warpReduceKernel(const float* data, float* results, int n, int warpSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    unsigned int mask = 0xffffffff; // 全 warp 活跃掩码
    // warp 内归约：每次将 offset 个线程后的数据累加到当前线程
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val += other;
    }

    // 将每个 warp 的归约结果写入 results，由线程 0 完成
    int laneId = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    if (laneId == 0) {
        int resIdx = blockIdx.x * (blockDim.x / warpSize) + warpId;
        results[resIdx] = val;
    }
}

int main() {
    const int N = 1024;
    const int threads = 128;
    const int blocks = 4;
    // 获取设备 warp 大小
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int warpSizeHost = prop.warpSize;
    const int warpsPerBlock = threads / warpSizeHost;

    size_t dataBytes = N * sizeof(float);
    size_t resBytes = blocks * warpsPerBlock * sizeof(float);

    // 初始化主机数据
    float* h_data = new float[N];
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f; // 全 1
    float* h_results = new float[blocks * warpsPerBlock];

    // 分配设备内存
    float *d_data, *d_results;
    cudaMalloc(&d_data, dataBytes);
    cudaMalloc(&d_results, resBytes);
    cudaMemcpy(d_data, h_data, dataBytes, cudaMemcpyHostToDevice);

    // 启动 kernel
    warpReduceKernel<<<blocks, threads>>>(d_data, d_results, N, warpSizeHost);
    cudaMemcpy(h_results, d_results, resBytes, cudaMemcpyDeviceToHost);

    // 打印每个 warp 的归约结果
    std::cout << "Warp reduction results (each warp sums): ";
    for (int i = 0; i < blocks * warpsPerBlock; ++i) {
        std::cout << h_results[i] << " ";
    }
    std::cout << std::endl;

    // 清理
    delete[] h_data;
    delete[] h_results;
    cudaFree(d_data);
    cudaFree(d_results);

    return 0;
} 