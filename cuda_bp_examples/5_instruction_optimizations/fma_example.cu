#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 检查 CUDA 错误的辅助宏
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(err_) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 核函数：演示 fused multiply-add (FMA) 指令
__global__ void fmaKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // fmaf(x, y, z) 在一条指令内完成乘加
        data[idx] = fmaf(data[idx], 2.0f, 3.0f);
    }
}

int main() {
    const int N = 1024 * 1024 * 10;
    const size_t bytes = N * sizeof(float);

    // 初始化主机数据
    std::vector<float> h_data(N, 0.0f);
    for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(i % 100) + 0.1f;

    // 分配设备内存并拷贝
    float *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // 执行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    fmaKernel<<<gridSize, blockSize>>>(d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "FMA (fmaf(val,2,3)): " << ms << " ms" << std::endl;

    // 清理
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
} 