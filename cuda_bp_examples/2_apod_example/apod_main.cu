#include <iostream>
#include <vector>
#include <numeric> // 用于 std::iota
#include <algorithm> // 用于 std::equal 和 std::generate
#include <chrono> // 用于基本计时 (可选，用于说明)
#include <cmath> // 用于 std::abs

// APOD Cycle (Assess, Parallelize, Optimize, Deploy) 教学示例
// 本代码演示 APOD 循环的前两个阶段：评估和并行化。
//
// 1. 评估 (Assess):
//    我们首先有一个在 CPU 上运行的函数 (vectorAddCPU)。
//    在实际应用中，我们会使用性能分析工具 (profiler) 来识别这类计算密集型函数作为"热点"。
//    这里，我们假设 vectorAddCPU 是一个已知需要优化的热点。
//
// 2. 并行化 (Parallelize):
//    我们将 vectorAddCPU 函数的功能移植到 GPU 上，使用 CUDA C++ 编写一个核函数 (vectorAddGPU_kernel)。
//    这是通过将计算任务分解给大量并行执行的 GPU 线程来完成的。
//
// (后续阶段，本示例未完全展开)
// 3. 优化 (Optimize):
//    在成功并行化之后，我们会进一步优化 GPU 核函数和数据传输，
//    以获得最佳性能 (参考 CUDA C++ Best Practices Guide 第 10-15 章)。
//    例如，考虑内存访问模式、指令优化、执行配置等。
//
// 4. 部署 (Deploy):
//    经过验证和优化后，将加速后的应用程序部署到生产环境 (参考指南第 16 章)。
//    APOD 是一个迭代过程，可能会多次重复以优化更多部分或进一步改进现有部分。

// CPU 版本的向量加法 (评估阶段的目标函数)
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// CUDA C++ 核函数 (并行化阶段的产物)
// __global__ 关键字表示这个函数是从 CPU 调用并在 GPU 上执行的。
// 它会被许多线程并行执行。
__global__ void vectorAddGPU_kernel(const float* a, const float* b, float* c, int n) {
    // 计算当前线程的全局唯一索引
    // threadIdx.x: 当前线程在块内的索引 (x维度)
    // blockIdx.x:  当前块在网格内的索引 (x维度)
    // blockDim.x:  每个块中的线程数量 (x维度)
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程索引在数组边界内
    // 因为我们可能会启动比元素数量更多的线程 (例如，如果 n 不是块大小的整数倍)
    if (index < n) {
        c[index] = a[index] + b[index]; // 每个线程执行一个元素的加法
    }
}

// 辅助函数，用于检查 CUDA 错误
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA 运行时错误: " << cudaGetErrorString(result) << std::endl;
    }
    return result;
}


int main() {
    std::cout << "APOD 周期教学示例: 向量加法\n";

    const int n = 1024 * 1024 * 4; // 定义向量大小 (例如 4M 个元素)
    std::cout << "向量大小: " << n << " 个元素\n";

    // --- 主机 (CPU) 数据准备 ---
    // 使用 std::vector 进行方便的内存管理
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c_cpu(n);    // CPU 计算结果
    std::vector<float> h_c_gpu(n);    // GPU 计算结果

    // 初始化输入向量 a 和 b
    // 这里使用 std::iota 和 std::generate 进行简单初始化
    std::iota(h_a.begin(), h_a.end(), 0.0f); // h_a = {0.0, 1.0, 2.0, ...}
    float start_val = 0.5f;
    std::generate(h_b.begin(), h_b.end(), [&start_val]() mutable { float current_val = start_val; start_val += 0.1f; return current_val; }); // h_b = {0.5, 0.6, 0.7, ...}

    // --- 1. 评估 (Assess) - 模拟执行 CPU 版本 ---
    std::cout << "\n--- CPU 执行 (评估阶段的基准) ---\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU 执行完成. 耗时: " << cpu_duration.count() << " ms\n";
    // 在实际评估中，我们会使用性能分析工具来确定 vectorAddCPU 这样的函数是瓶颈。

    // --- 2. 并行化 (Parallelize) - 执行 GPU 版本 ---
    std::cout << "\n--- GPU 执行 (并行化阶段) ---\n";

    // 设备 (GPU) 内存指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 为设备分配内存
    // cudaMalloc: 在 GPU 全局内存中分配指定大小的内存
    std::cout << "正在分配 GPU 内存...\n";
    if (checkCuda(cudaMalloc((void**)&d_a, n * sizeof(float))) != cudaSuccess) return 1;
    if (checkCuda(cudaMalloc((void**)&d_b, n * sizeof(float))) != cudaSuccess) { cudaFree(d_a); return 1; }
    if (checkCuda(cudaMalloc((void**)&d_c, n * sizeof(float))) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return 1; }
    std::cout << "GPU 内存分配成功.\n";

    auto start_gpu_total = std::chrono::high_resolution_clock::now();

    // 将数据从主机内存 (h_a, h_b) 复制到设备内存 (d_a, d_b)
    // cudaMemcpyHostToDevice: 指定从主机到设备的内存复制
    std::cout << "正在将数据从主机复制到 GPU...\n";
    auto start_memcpy_h2d = std::chrono::high_resolution_clock::now();
    if (checkCuda(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { /* 清理 */ cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 1; }
    if (checkCuda(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { /* 清理 */ cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 1; }
    auto end_memcpy_h2d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memcpy_h2d_duration = end_memcpy_h2d - start_memcpy_h2d;
    std::cout << "数据从主机复制到 GPU 完成。耗时: " << memcpy_h2d_duration.count() << " ms\n";


    // 定义核函数执行配置
    // 线程块大小 (threads per block)
    int threadsPerBlock = 256;
    // 网格大小 (blocks per grid)
    // 需要足够的块来覆盖所有 n 个元素
    // (n + threadsPerBlock - 1) / threadsPerBlock 是一种计算向上取整的常用方法
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "启动 CUDA 核函数，配置: " << blocksPerGrid << " 个块, " << threadsPerBlock << " 个线程/块\n";

    // 启动核函数
    // <<<blocksPerGrid, threadsPerBlock>>> 是 CUDA 核函数的启动语法
    auto start_kernel = std::chrono::high_resolution_clock::now();
    vectorAddGPU_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    if (checkCuda(cudaGetLastError()) != cudaSuccess) { /* 清理 */ cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 1; } // 检查核函数启动是否出错

    // 等待 GPU 完成所有之前提交的任务 (包括核函数执行)
    // 核函数启动是异步的，所以需要同步来确保结果可用和准确计时
    if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess) { /* 清理 */ cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 1; }
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_duration = end_kernel - start_kernel;
    std::cout << "GPU 核函数执行完成。耗时: " << kernel_duration.count() << " ms\n";

    // 将结果从设备内存 (d_c) 复制回主机内存 (h_c_gpu)
    // cudaMemcpyDeviceToHost: 指定从设备到主机的内存复制
    std::cout << "正在将结果从 GPU 复制回主机...\n";
    auto start_memcpy_d2h = std::chrono::high_resolution_clock::now();
    if (checkCuda(cudaMemcpy(h_c_gpu.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) { /* 清理 */ cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 1; }
    auto end_memcpy_d2h = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memcpy_d2h_duration = end_memcpy_d2h - start_memcpy_d2h;
    std::cout << "结果从 GPU 复制回主机完成。耗时: " << memcpy_d2h_duration.count() << " ms\n";

    auto end_gpu_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_total_duration = end_gpu_total - start_gpu_total;
    std::cout << "GPU 总执行时间 (包括内存复制和核函数): " << gpu_total_duration.count() << " ms\n";

    // --- 验证结果 (可选但重要) ---
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) { // 比较浮点数，允许一些误差
            std::cerr << "结果不匹配! 索引 " << i << ": CPU=" << h_c_cpu[i] << ", GPU=" << h_c_gpu[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "\n结果验证成功: CPU 和 GPU 版本输出一致.\n";
    } else {
        std::cout << "\n结果验证失败!\n";
    }

    // --- 清理 GPU 内存 ---
    // cudaFree: 释放之前通过 cudaMalloc 分配的设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::cout << "GPU 内存已释放.\n";

    // --- 3. 优化 (Optimize) ---
    std::cout << "\n--- 优化阶段 (概念) ---\n";
    std::cout << "在并行化之后，下一步是对 GPU 实现进行优化.\n";
    std::cout << "这可能包括:\n";
    std::cout << " - 优化内存访问模式 (例如，使用共享内存，确保合并访问).\n";
    std::cout << " - 调整核函数启动配置 (线程块大小，网格大小).\n";
    std::cout << " - 使用 CUDA Streams 来重叠数据传输和计算.\n";
    std::cout << " - 减少主机和设备之间的数据传输.\n";
    std::cout << " - 指令级优化 (例如，使用更快的数学函数).\n";
    std::cout << "参考 CUDA C++ Best Practices Guide 第 10-15 章获取详细信息.\n";

    // --- 4. 部署 (Deploy) ---
    std::cout << "\n--- 部署阶段 (概念) ---\n";
    std::cout << "一旦优化完成并通过验证，加速后的应用就可以部署到生产环境了.\n";
    std::cout << "APOD 周期可以根据需要重复进行，以进一步提升性能或加速其他部分.\n";
    std::cout << "参考 CUDA C++ Best Practices Guide 第 16 章获取部署相关信息.\n";

    return 0;
} 