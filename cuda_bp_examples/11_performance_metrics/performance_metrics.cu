#include <iostream>
#include <chrono>      // 用于 CPU 计时
#include <cuda_runtime.h>
#include <cstring>     // 用于 memcpy

// 检查 CUDA 错误的辅助函数
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA 错误: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

// 向量加法核函数：c[i] = a[i] + b[i]
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "=== CUDA 性能度量示例 ===" << std::endl;
    // 教学要点：
    // 1) 演示如何使用 std::chrono 在 CPU 端对内存拷贝和核函数执行进行计时
    //    需要在异步操作前后进行同步，以确保时间度量准确。
    // 2) 演示如何使用 CUDA 事件 (cudaEvent_t) 在 GPU 端对核函数执行进行高精度测量，
    //    其时间不受 CPU 时间戳误差影响，但只能测设备端执行。
    // 3) 计算并打印主机到设备 Host->Device 传输带宽，以及核函数的有效带宽，
    //    帮助评估内存访问效率和核函数性能。

    // 向量大小
    const int N = 1 << 20; // 约 1M 个元素
    const size_t bytes = N * sizeof(float);

    // 分配主机内存
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i) * 0.5f;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, bytes));
    checkCuda(cudaMalloc(&d_b, bytes));
    checkCuda(cudaMalloc(&d_c, bytes));

    // --- 使用 CPU 计时器度量 Host->Device 传输 ---
    // 教学：使用 CPU 计时器直接记录 cudaMemcpy 同步执行时间，
    //      统计从主机到设备两次拷贝的总耗时，并计算带宽 = 数据量 / 时间。
    auto start_h2d = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    auto end_h2d = std::chrono::high_resolution_clock::now();
    float time_h2d = std::chrono::duration<float, std::milli>(end_h2d - start_h2d).count();
    float bw_h2d = (2 * bytes / 1e9f) / (time_h2d / 1000.0f);
    std::cout << "Host->Device 时间 (CPU 计时): " << time_h2d << " ms, 带宽: "
              << bw_h2d << " GB/s" << std::endl;

    // 核函数配置
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // --- 使用 CPU 计时器度量核函数执行时间 ---
    // 教学：同样使用 CPU 计时器测量 kernel 启动到同步完成的时间，
    //      包含内核调度和执行开销，但会受 CPU→GPU 同步误差影响。
    auto start_kernel_cpu = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    checkCuda(cudaDeviceSynchronize());
    auto end_kernel_cpu = std::chrono::high_resolution_clock::now();
    float time_kernel_cpu = std::chrono::duration<float, std::milli>(end_kernel_cpu - start_kernel_cpu).count();
    std::cout << "核函数执行时间 (CPU 计时): " << time_kernel_cpu << " ms" << std::endl;

    // --- 使用 CUDA 事件度量核函数执行时间 ---
    // 教学：CUDA 事件记录的是 GPU 时间戳，只在 GPU 端记录事件前后，
    //      精度更高且不依赖主机同步。但仅能测量设备内部执行，不含内存拷贝。
    cudaEvent_t ev_start, ev_stop;
    checkCuda(cudaEventCreate(&ev_start));
    checkCuda(cudaEventCreate(&ev_stop));
    checkCuda(cudaEventRecord(ev_start, 0));
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    checkCuda(cudaEventRecord(ev_stop, 0));
    checkCuda(cudaEventSynchronize(ev_stop));
    float time_kernel_ev = 0;
    checkCuda(cudaEventElapsedTime(&time_kernel_ev, ev_start, ev_stop));
    std::cout << "核函数执行时间 (CUDA 事件): " << time_kernel_ev << " ms" << std::endl;

    // 计算有效带宽：
    // 教学：有效带宽 = (读数据 2 × bytes + 写数据 1 × bytes) / 核函数执行时间，
    //      反映核函数对设备内存的吞吐效率，与理论峰值比较可找出内存瓶颈。
    float bytes_accessed = 3 * bytes;
    float bw_effective = (bytes_accessed / 1e9f) / (time_kernel_ev / 1000.0f);
    std::cout << "核函数有效带宽: " << bw_effective << " GB/s" << std::endl;

    // --- 结果和带宽输出解读 ---
    // 教学：示例输出：
    //  Host->Device 时间: ~1.7 ms, 带宽: ~4.9 GB/s （受限于 PCIe）
    //  核函数执行时间 (CUDA 事件): ~0.036 ms, 有效带宽: ~340 GB/s （接近 HBM 理论带宽）
    //  通过对比可判断拷贝和计算的瓶颈所在，并指导后续优化方向。
    checkCuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    std::cout << "验证: c[0] = " << h_c[0] << ", c[N-1] = " << h_c[N-1] << std::endl;

    // --- 异步多流流水线示例: 隐藏拷贝延迟 ---
    const int nStream = 2;
    const int chunkSize = N / nStream;
    const size_t chunkBytes = chunkSize * sizeof(float);
    std::cout << "\n--- 异步多流流水线示例 ---" << std::endl;
    // 分配页锁定主机内存
    float *hp_a, *hp_b, *hp_c;
    checkCuda(cudaHostAlloc(&hp_a, bytes, cudaHostAllocDefault));
    checkCuda(cudaHostAlloc(&hp_b, bytes, cudaHostAllocDefault));
    checkCuda(cudaHostAlloc(&hp_c, bytes, cudaHostAllocDefault));
    // 将原始数据复制到页锁定内存
    std::memcpy(hp_a, h_a, bytes);
    std::memcpy(hp_b, h_b, bytes);
    // 分配每个流的设备缓冲区并创建 CUDA 流
    float *d_a_s[nStream], *d_b_s[nStream], *d_c_s[nStream];
    cudaStream_t streams[nStream];
    for (int i = 0; i < nStream; ++i) {
        checkCuda(cudaMalloc(&d_a_s[i], chunkBytes));
        checkCuda(cudaMalloc(&d_b_s[i], chunkBytes));
        checkCuda(cudaMalloc(&d_c_s[i], chunkBytes));
        checkCuda(cudaStreamCreate(&streams[i]));
    }
    // 测量流水线执行时间
    auto start_pipe = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nStream; ++i) {
        int offset = i * chunkSize;
        // 异步拷贝到设备 + 核函数 + 异步拷回
        checkCuda(cudaMemcpyAsync(d_a_s[i], hp_a + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        checkCuda(cudaMemcpyAsync(d_b_s[i], hp_b + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        vectorAddKernel<<<(chunkSize + threads - 1) / threads, threads, 0, streams[i]>>>(
            d_a_s[i], d_b_s[i], d_c_s[i], chunkSize);
        checkCuda(cudaMemcpyAsync(hp_c + offset, d_c_s[i], chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    // 同步所有流
    for (int i = 0; i < nStream; ++i) {
        checkCuda(cudaStreamSynchronize(streams[i]));
    }
    auto end_pipe = std::chrono::high_resolution_clock::now();
    float time_pipe = std::chrono::duration<float, std::milli>(end_pipe - start_pipe).count();
    // 有效带宽同样按 3 倍数据量计算
    float bw_pipe = (3 * bytes / 1e9f) / (time_pipe / 1000.0f);
    std::cout << "多流流水线总耗时: " << time_pipe << " ms, 带宽: "
              << bw_pipe << " GB/s" << std::endl;
    // 验证多流结果
    bool pipeline_ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(hp_c[i] - h_c[i]) > 1e-5f) { pipeline_ok = false; break; }
    }
    std::cout << "多流流水线结果 " << (pipeline_ok ? "正确" : "错误") << std::endl;
    // 清理多流资源
    for (int i = 0; i < nStream; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a_s[i]); cudaFree(d_b_s[i]); cudaFree(d_c_s[i]);
    }
    cudaFreeHost(hp_a); cudaFreeHost(hp_b); cudaFreeHost(hp_c);

    // 清理资源
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));
    checkCuda(cudaEventDestroy(ev_start));
    checkCuda(cudaEventDestroy(ev_stop));

    return 0;
} 