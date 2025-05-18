#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vectorization 示例
// 教学：使用 float4 一次加载 4 个 float，减少内存事务次数，提升带宽利用率

// 标量拷贝核函数：每个线程拷贝一个 float
__global__ void scalarCopyKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// 向量化拷贝核函数：每个线程拷贝一个 float4
__global__ void vectorCopyKernel(const float4* in, float4* out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        out[idx] = in[idx];
    }
}

int main() {
    // 元素数量须为 4 的倍数
    const int N = 1 << 22; // 4M floats (~16 MB)
    const int N4 = N / 4;
    size_t bytes = N * sizeof(float);
    size_t bytes4 = N4 * sizeof(float4);

    // 分配页锁定主机内存
    float* h_in;
    float* h_out;
    cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_out, bytes, cudaHostAllocDefault);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // 分配设备内存
    float *d_in_scalar, *d_out_scalar;
    float4 *d_in_vec, *d_out_vec;
    cudaMalloc(&d_in_scalar, bytes);
    cudaMalloc(&d_out_scalar, bytes);
    cudaMalloc(&d_in_vec, bytes4);
    cudaMalloc(&d_out_vec, bytes4);

    // 拷贝输入数据到设备
    cudaMemcpy(d_in_scalar, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_vec, h_in, bytes, cudaMemcpyHostToDevice); // reinterpret as float4*

    // 执行配置
    int blockSize = 256;
    int gridScalar = (N + blockSize - 1) / blockSize;
    int gridVec    = (N4 + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float msScalar = 0.0f, msVec = 0.0f;

    // 标量拷贝测时
    cudaEventRecord(start);
    scalarCopyKernel<<<gridScalar, blockSize>>>(d_in_scalar, d_out_scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msScalar, start, stop);

    // 向量化拷贝测时
    cudaEventRecord(start);
    vectorCopyKernel<<<gridVec, blockSize>>>(d_in_vec, d_out_vec, N4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msVec, start, stop);

    // 输出结果
    std::cout << "Scalar copy time: " << msScalar << " ms, bandwidth: "
              << (bytes / 1e9f) / (msScalar / 1000.0f) << " GB/s" << std::endl;
    std::cout << "Vector copy time: " << msVec << " ms, bandwidth: "
              << (bytes / 1e9f) / (msVec / 1000.0f) << " GB/s" << std::endl;

    // 清理
    cudaFree(d_in_scalar);
    cudaFree(d_out_scalar);
    cudaFree(d_in_vec);
    cudaFree(d_out_vec);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} 