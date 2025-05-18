#include <iostream>
#include <vector>
#include <chrono>
#include <cstring> // For std::memcpy
#include <cuda_runtime.h>
#include <string>

// 检查 CUDA 错误的辅助函数
inline cudaError_t checkCuda(cudaError_t result, const char* func_name = nullptr) {
    if (result != cudaSuccess) {
        if (func_name) {
            std::cerr << "CUDA Error at " << func_name << ": ";
        }
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

// 简单的核函数，用于确保 GPU 有工作要做
__global__ void dummyKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// 测量数据传输带宽的函数
// 教学：支持 4 种模式（同步/异步、H2D/D2H）
//       使用 CUDA 事件测量执行时间，并取 10 次平均降低抖动
void benchmarkTransfer(const std::string& test_name, const float* h_data, float* d_data, size_t bytes, 
                       cudaMemcpyKind kind, cudaStream_t stream = 0, bool is_async = false) 
{
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    float ms_transfer = 0.0f;
    cudaEvent_t start_event, stop_event;
    checkCuda(cudaEventCreate(&start_event));
    checkCuda(cudaEventCreate(&stop_event));

    // 预热拷贝，避免首次拷贝的额外开销影响测量
    if (is_async) {
        if (kind == cudaMemcpyHostToDevice) {
            checkCuda(cudaMemcpyAsync(d_data, h_data, bytes, kind, stream), "cudaMemcpyAsync H2D");
        } else {
            // DeviceToHost 反向拷贝需交换 dst/src
            checkCuda(cudaMemcpyAsync((void*)h_data, d_data, bytes, kind, stream), "cudaMemcpyAsync D2H");
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } else {
        if (kind == cudaMemcpyHostToDevice) {
            checkCuda(cudaMemcpy(d_data, h_data, bytes, kind), "cudaMemcpy H2D");
        } else {
            // DeviceToHost 同步拷贝
            checkCuda(cudaMemcpy((void*)h_data, d_data, bytes, kind), "cudaMemcpy D2H");
        }
    }

    // 实际测量
    checkCuda(cudaEventRecord(start_event, stream), "cudaEventRecord start");
    for (int i = 0; i < 10; ++i) { // 多次拷贝取平均，减少波动
        if (is_async) {
            if (kind == cudaMemcpyHostToDevice) {
                checkCuda(cudaMemcpyAsync(d_data, h_data, bytes, kind, stream), "cudaMemcpyAsync H2D");
            } else {
                checkCuda(cudaMemcpyAsync((void*)h_data, d_data, bytes, kind, stream), "cudaMemcpyAsync D2H");
            }
        } else {
            if (kind == cudaMemcpyHostToDevice) {
                checkCuda(cudaMemcpy(d_data, h_data, bytes, kind), "cudaMemcpy H2D");
            } else {
                checkCuda(cudaMemcpy((void*)h_data, d_data, bytes, kind), "cudaMemcpy D2H");
            }
        }
    }
    checkCuda(cudaEventRecord(stop_event, stream), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop_event)); // 对于异步拷贝，确保所有操作完成
    checkCuda(cudaEventElapsedTime(&ms_transfer, start_event, stop_event));

    ms_transfer /= 10.0f; // 平均时间
    float gb_per_sec = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (ms_transfer / 1000.0f);
    std::cout << "Transfer Time: " << ms_transfer << " ms" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;

    checkCuda(cudaEventDestroy(start_event));
    checkCuda(cudaEventDestroy(stop_event));
}

int main() {
    const int N = 1 << 22; // 4M elements -> 16MB
    const size_t bytes = N * sizeof(float);

    // 教学要点：对比四种主机内存策略
    // 1) Pageable Host Memory：CPU 可分页内存，默认 new/malloc，仅支持同步拷贝，带宽最低
    // 2) Pinned Default：cudaHostAllocDefault，支持异步拷贝，比 pageable 稍快，兼顾缓存
    // 3) Pinned WriteCombined：cudaHostAllocWriteCombined，优化 H->D 流式写入，不走 CPU 缓存
    // 4) Zero Copy：cudaHostAllocMapped + cudaHostGetDevicePointer，GPU 直接访问主机内存，适合小量/随机访问

    std::cout << "=== Pinned Memory Bandwidth Test ===" << std::endl;
    std::cout << "Data size: " << bytes / (1024.0f * 1024.0f) << " MB" << std::endl;

    // 设备缓冲
    float *d_buffer1, *d_buffer2;
    checkCuda(cudaMalloc(&d_buffer1, bytes), "cudaMalloc d_buffer1");
    checkCuda(cudaMalloc(&d_buffer2, bytes), "cudaMalloc d_buffer2");

    // 1. 标准可分页主机内存 (Pageable Host Memory)
    // 教学：new 分配的内存不能用于异步 DMA，只能同步拷贝，带宽受限 ~5-6GB/s
    float* h_pageable = new float[N];
    for (int i = 0; i < N; ++i) h_pageable[i] = static_cast<float>(i);
    benchmarkTransfer("Pageable H->D (sync)", h_pageable, d_buffer1, bytes, cudaMemcpyHostToDevice);
    benchmarkTransfer("Pageable D->H (sync)", h_pageable, d_buffer1, bytes, cudaMemcpyDeviceToHost);
    delete[] h_pageable;

    // 2. 页锁定主机内存 (Pinned Host Memory - Default)
    // 教学：cudaHostAllocDefault 分配可缓存页锁定内存，支持异步 DMA，带宽可略增
    float* h_pinned_default;
    checkCuda(cudaHostAlloc(&h_pinned_default, bytes, cudaHostAllocDefault), "cudaHostAlloc Default");
    for (int i = 0; i < N; ++i) h_pinned_default[i] = static_cast<float>(i);
    benchmarkTransfer("Pinned Default H->D (sync)", h_pinned_default, d_buffer1, bytes, cudaMemcpyHostToDevice);
    benchmarkTransfer("Pinned Default D->H (sync)", h_pinned_default, d_buffer1, bytes, cudaMemcpyDeviceToHost);

    // 3. Pinned Default + Async Copy
    // 教学：结合 cudaMemcpyAsync 和 stream，可隐藏部分传输延迟
    cudaStream_t stream_default_async;
    checkCuda(cudaStreamCreate(&stream_default_async), "cudaStreamCreate for Default Pinned");
    benchmarkTransfer("Pinned Default H->D (async)", h_pinned_default, d_buffer1, bytes, cudaMemcpyHostToDevice, stream_default_async, true);
    benchmarkTransfer("Pinned Default D->H (async)", h_pinned_default, d_buffer1, bytes, cudaMemcpyDeviceToHost, stream_default_async, true);
    checkCuda(cudaStreamDestroy(stream_default_async), "cudaStreamDestroy for Default Pinned");
    checkCuda(cudaFreeHost(h_pinned_default), "cudaFreeHost Default");

    // 4. 页锁定主机内存 (WriteCombined)
    // 教学：cudaHostAllocWriteCombined 分配写组合内存，适用于大量连续 H->D 写入，不会缓存至 CPU L1/L2
    float* h_pinned_wc;
    checkCuda(cudaHostAlloc(&h_pinned_wc, bytes, cudaHostAllocWriteCombined), "cudaHostAlloc WriteCombined");
    for (int i = 0; i < N; ++i) h_pinned_wc[i] = static_cast<float>(i);
    benchmarkTransfer("Pinned WriteCombined H->D (sync)", h_pinned_wc, d_buffer1, bytes, cudaMemcpyHostToDevice);
    benchmarkTransfer("Pinned WriteCombined D->H (sync)", h_pinned_wc, d_buffer1, bytes, cudaMemcpyDeviceToHost);

    // 5. WriteCombined + Async Copy
    cudaStream_t stream_wc_async;
    checkCuda(cudaStreamCreate(&stream_wc_async), "cudaStreamCreate for WC Pinned");
    benchmarkTransfer("Pinned WriteCombined H->D (async)", h_pinned_wc, d_buffer1, bytes, cudaMemcpyHostToDevice, stream_wc_async, true);
    benchmarkTransfer("Pinned WriteCombined D->H (async)", h_pinned_wc, d_buffer1, bytes, cudaMemcpyDeviceToHost, stream_wc_async, true);
    checkCuda(cudaStreamDestroy(stream_wc_async), "cudaStreamDestroy for WC Pinned");
    checkCuda(cudaFreeHost(h_pinned_wc), "cudaFreeHost WriteCombined");

    // 6. Zero-copy (Mapped Pinned Memory)
    // 教学：cudaHostAllocMapped + cudaHostGetDevicePointer 实现 GPU 直接访问主机内存，
    //       但不走设备缓存，带宽低 (~1GB/s)，适用于少量或随机访问。
    float *h_mapped = nullptr, *d_mapped = nullptr;
    checkCuda(cudaHostAlloc(&h_mapped, bytes, cudaHostAllocMapped), "cudaHostAlloc Mapped");
    for (int i = 0; i < N; ++i) h_mapped[i] = static_cast<float>(i);
    checkCuda(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0), "cudaHostGetDevicePointer");
    // 事件计时核函数读取主机映射内存
    cudaEvent_t zstart, zstop;
    checkCuda(cudaEventCreate(&zstart));
    checkCuda(cudaEventCreate(&zstop));
    checkCuda(cudaEventRecord(zstart, 0));
    dummyKernel<<<(N+255)/256,256>>>(d_mapped, d_buffer2, N);
    checkCuda(cudaEventRecord(zstop, 0));
    checkCuda(cudaEventSynchronize(zstop));
    float tz = 0; checkCuda(cudaEventElapsedTime(&tz, zstart, zstop));
    float bw_z = (bytes / 1e9f) / (tz / 1000.0f);
    std::cout << "Zero-copy Kernel Time: " << tz << " ms, 带宽: " << bw_z << " GB/s" << std::endl;
    checkCuda(cudaEventDestroy(zstart)); checkCuda(cudaEventDestroy(zstop));
    checkCuda(cudaFreeHost(h_mapped), "cudaFreeHost Mapped");

    // 清理设备内存
    checkCuda(cudaFree(d_buffer1), "cudaFree d_buffer1");
    checkCuda(cudaFree(d_buffer2), "cudaFree d_buffer2");

    std::cout << "\nTest finished." << std::endl;
    return 0;
} 