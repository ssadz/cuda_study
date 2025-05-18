#include <iostream>
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

// 标准加载核函数：每个线程加载一个 float
__global__ void normalLoadKernel(const float* __restrict__ in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// __ldg 加载核函数：每个线程使用只读缓存加载一个 float
__global__ void ldgLoadKernel(const float* __restrict__ in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __ldg(&in[idx]);
    }
}

int main() {
    const int N = 1 << 22; // 4M elements (~16 MB)
    size_t bytes = N * sizeof(float);

    // 分配页锁定主机内存并初始化
    float *h_in, *h_out;
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault));
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // 分配设备内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 执行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float msNormal = 0.0f, msLDG = 0.0f;

    // 普通加载测时
    CUDA_CHECK(cudaEventRecord(start));
    normalLoadKernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msNormal, start, stop));

    // __ldg 加载测时
    CUDA_CHECK(cudaEventRecord(start));
    ldgLoadKernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msLDG, start, stop));

    // 输出带宽对比
    std::cout << "Normal load time: " << msNormal << " ms, bandwidth: "
              << (bytes / 1e9f) / (msNormal / 1000.0f) << " GB/s" << std::endl;
    std::cout << "__ldg load time: " << msLDG << " ms, bandwidth: "
              << (bytes / 1e9f) / (msLDG / 1000.0f) << " GB/s" << std::endl;

    // 清理
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
} 