#include <iostream>
#include <vector>
#include <cmath>         // 用于 sin、sqrt
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

// 核函数：演示双精度数学函数
__global__ void doublePrecisionKernel(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = data[idx];
        data[idx] = sin(val) + sqrt(val); // 双精度标准库函数
    }
}

int main() {
    const int N = 1024 * 1024 * 10;
    const size_t bytes = N * sizeof(double);

    // 初始化主机数据
    std::vector<double> h_data(N, 0.0);
    for (int i = 0; i < N; ++i) h_data[i] = static_cast<double>(i % 100) + 0.1;

    // 分配设备内存并拷贝
    double *d_data, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // 设置执行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    doublePrecisionKernel<<<gridSize, blockSize>>>(d_result, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Double Precision (sin + sqrt): " << ms << " ms" << std::endl;

    // 清理
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
} 