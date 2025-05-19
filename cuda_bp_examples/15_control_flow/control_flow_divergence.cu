#include <iostream>
#include <vector>
#include <numeric>      // 用于 std::iota
#include <algorithm>    // 用于 std::all_of
#include <chrono>
#include <cmath>        // 用于 std::abs

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // 用于 warpSize

// 辅助函数：检查 CUDA API 调用结果
inline cudaError_t checkCuda(cudaError_t result, const char* func_name = nullptr) {
    if (result != cudaSuccess) {
        if (func_name) {
            // 字符串字面量保持英文
            std::cerr << "CUDA Error at " << func_name << ": ";
        }
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

// 核函数 1: 存在潜在的 Warp 发散 (线程根据自身 ID 的奇偶性选择不同路径)
// 教学: 演示当 warp 内的线程执行不同指令路径时，性能会下降，因为路径需要串行化执行。
__global__ void divergentBranchKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) { // 偶数索引线程执行的操作
            data[idx] = data[idx] * 2.0f;
            data[idx] = data[idx] + 1.0f;
        } else { // 奇数索引线程执行的操作 (不同于偶数)
            data[idx] = data[idx] / 2.0f;
            data[idx] = data[idx] - 1.0f;
        }
    }
}

// 核函数 2: 尝试避免 Warp 发散 (所有线程执行相同路径，通过计算选择值)
// 教学: 通过计算而不是分支来决定操作数，可以减少或避免发散。
//       虽然可能增加了少量计算，但在某些情况下，避免发散的收益更大。
__global__ void coalescedBranchKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_even = (idx % 2 == 0);
        float val = data[idx];

        // 使用条件赋值或计算来模拟分支逻辑，而不是真正的 if-else 路径分离
        // 这种方式下，所有线程执行相同的指令序列
        float multiplied_val = val * 2.0f;
        float divided_val = val / 2.0f;
        float added_val = multiplied_val + 1.0f;
        float subtracted_val = divided_val - 1.0f;

        data[idx] = is_even ? added_val : subtracted_val;
    }
}

// 核函数 3: Warp 内统一路径 (如果条件对于整个 Warp 是一致的，则不会发散)
// 教学: 演示如果分支条件对于一个 warp 内的所有线程结果都相同，则该 warp 不会发散。
__global__ void warpUniformBranchKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // warp_id_in_block 计算: threadIdx.x / warpSize (warpSize 是 CUDA 内置变量)
    // 这里我们假设 blockDim.x 是 warpSize 的整数倍，以便简化 warp_id_in_block 的计算和演示
    int warp_id_in_block = threadIdx.x / warpSize;

    if (idx < n) {
        // 条件基于 warp ID，因此一个 warp 内的所有线程会走相同路径
        if (warp_id_in_block % 2 == 0) {
            data[idx] = data[idx] * 3.0f;
            data[idx] = data[idx] - 5.0f;
        } else {
            data[idx] = data[idx] * 0.5f;
            data[idx] = data[idx] + 5.0f;
        }
    }
}


int main() {
    // 字符串字面量保持英文
    std::cout << "CUDA Control Flow - Branch Divergence Example (Best Practices Chapter 15)" << std::endl;

    const int N = 1024 * 1024 * 8; // 元素数量
    const size_t dataSize = N * sizeof(float); // 数据总字节数

    // --- 主机数据准备 ---
    std::vector<float> h_data_original(N);
    std::vector<float> h_data_divergent(N);
    std::vector<float> h_data_coalesced(N);
    std::vector<float> h_data_warp_uniform(N);
    std::vector<float> h_data_expected_divergent(N);
    std::vector<float> h_data_expected_coalesced(N);
    std::vector<float> h_data_expected_warp_uniform(N);


    // 初始化原始数据
    std::iota(h_data_original.begin(), h_data_original.end(), 1.0f); // {1.0f, 2.0f, ...}

    // 计算期望结果 (用于验证)
    for(int i = 0; i < N; ++i) {
        float val = h_data_original[i];
        if (i % 2 == 0) {
            h_data_expected_divergent[i] = val * 2.0f + 1.0f;
        } else {
            h_data_expected_divergent[i] = val / 2.0f - 1.0f;
        }
        // coalescedBranchKernel 应该与 divergentBranchKernel 得到相同的结果
        h_data_expected_coalesced[i] = h_data_expected_divergent[i];

        // warpUniformBranchKernel 的期望结果计算
        // 需要模拟核函数内 warp_id_in_block 的逻辑
        // 注意：这里的 threadsPerBlock 和 warpSize 需要与核函数启动配置和硬件特性匹配
        int threadsPerBlock_cpu = 256; // 与核函数启动配置一致
        int warpSize_cpu = 32;       // 标准 warp 大小
        // int block_id_cpu = i / threadsPerBlock_cpu; // 当前元素属于哪个块
        int thread_in_block_id_cpu = i % threadsPerBlock_cpu; // 当前元素在块内的线程索引
        int warp_id_in_block_cpu = thread_in_block_id_cpu / warpSize_cpu; // 当前元素所在 warp 在其块内的 ID

        if (warp_id_in_block_cpu % 2 == 0) {
             h_data_expected_warp_uniform[i] = val * 3.0f - 5.0f;
        } else {
             h_data_expected_warp_uniform[i] = val * 0.5f + 5.0f;
        }
    }


    // --- 设备数据 ---
    float* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, dataSize), "cudaMalloc d_data");

    // CUDA 执行配置
    int threadsPerBlock = 256; // 确保是 warpSize (通常是32) 的倍数
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 计算网格大小

    cudaEvent_t start_event, stop_event;
    checkCuda(cudaEventCreate(&start_event));
    checkCuda(cudaEventCreate(&stop_event));
    float ms; // 用于存储耗时（毫秒）

    // --- 1. 测试 Divergent Branch Kernel ---
    std::cout << "\n--- Testing Divergent Branch Kernel ---" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for divergent test");

    checkCuda(cudaEventRecord(start_event));
    divergentBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "divergentBranchKernel launch check"); // 检查核函数启动错误
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize for divergent test"); // 等待核函数完成
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event)); // 计算耗时
    // 字符串字面量保持英文
    std::cout << "Divergent Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_divergent.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for divergent test");
    bool divergent_ok = true;
    for(int i=0; i<N; ++i) {
        if(std::abs(h_data_divergent[i] - h_data_expected_divergent[i]) > 1e-4) { // 允许浮点误差
            divergent_ok = false;
            // 字符串字面量保持英文
            // std::cerr << "Verification failed for divergentBranchKernel at index " << i << std::endl;
            break;
        }
    }
    // 字符串字面量保持英文
    std::cout << "Divergent Branch Kernel Verification: " << (divergent_ok ? "PASSED" : "FAILED") << std::endl;


    // --- 2. 测试 Coalesced Branch Kernel (模拟分支) ---
    // 字符串字面量保持英文
    std::cout << "\n--- Testing Coalesced Branch Kernel (emulating branch) ---" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for coalesced test");

    checkCuda(cudaEventRecord(start_event));
    coalescedBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "coalescedBranchKernel launch check");
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize for coalesced test");
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "Coalesced Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_coalesced.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for coalesced test");
    bool coalesced_ok = true;
    for(int i=0; i<N; ++i) {
        if(std::abs(h_data_coalesced[i] - h_data_expected_coalesced[i]) > 1e-4) {
            coalesced_ok = false;
            // 字符串字面量保持英文
            // std::cerr << "Verification failed for coalescedBranchKernel at index " << i << std::endl;
            break;
        }
    }
    // 字符串字面量保持英文
    std::cout << "Coalesced Branch Kernel Verification: " << (coalesced_ok ? "PASSED" : "FAILED") << std::endl;

    // --- 3. 测试 Warp Uniform Branch Kernel ---
    // 字符串字面量保持英文
    std::cout << "\n--- Testing Warp Uniform Branch Kernel ---" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for warp_uniform test");

    checkCuda(cudaEventRecord(start_event));
    warpUniformBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "warpUniformBranchKernel launch check");
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize for warp_uniform test");
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "Warp Uniform Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_warp_uniform.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for warp_uniform test");
    bool warp_uniform_ok = true;
    for(int i=0; i<N; ++i) {
        if(std::abs(h_data_warp_uniform[i] - h_data_expected_warp_uniform[i]) > 1e-4) {
            // 字符串字面量保持英文
            // std::cerr << "Verification failed for warpUniformBranchKernel at index " << i
            //           << ": Got " << h_data_warp_uniform[i] << ", Expected " << h_data_expected_warp_uniform[i] << std::endl;
            warp_uniform_ok = false;
            // break; // 如果需要查看所有错误，可以注释掉 break
        }
    }
    // 字符串字面量保持英文
    std::cout << "Warp Uniform Branch Kernel Verification: " << (warp_uniform_ok ? "PASSED" : "FAILED") << std::endl;


    // 字符串字面量保持英文
    //std::cout << "\nTeaching Points (Best Practices Guide - Chapter 15.1 Branching and Divergence):" << std::endl;
    //// 中文注释开始
    //std::cout << "1. divergentBranchKernel: 当 warp 内的线程根据各自的 `idx % 2` 执行不同代码路径时，会发生分支发散。" << std::endl;
    //std::cout << "   这通常会导致性能下降，因为不同的路径需要被硬件串行化执行。" << std::endl;
    //std::cout << "2. coalescedBranchKernel: 尝试通过计算（例如使用三元操作符）来选择最终值，而不是创建两个截然不同的执行路径。" << std::endl;
    //std::cout << "   目标是让所有线程执行相同的指令序列，尽管它们操作的数据可能不同或最终结果不同。" << std::endl;
    //std::cout << "   这种方法是否比发散分支更快，取决于具体计算的复杂度和分支的代价。" << std::endl;
    //std::cout << "3. warpUniformBranchKernel: 当分支条件对于一个 warp 内的所有线程结果都相同时（例如，条件基于 warp ID），" << std::endl;
    //std::cout << "   那么该 warp 内的线程将走同一路径，不会发生 warp 内发散，从而保持较好的性能。" << std::endl;
    //std::cout << "4. 实际性能差异取决于 GPU 架构、编译器优化以及分支内代码的复杂度。" << std::endl;
    //std::cout << "   通常，应尽量避免或减少不必要的 warp 发散。" << std::endl;
    //std::cout << "   如果发散不可避免，尝试使发散的代码段尽可能短。" << std::endl;
    //std::cout << "   现代 GPU (如 Volta 及之后架构) 具有独立线程调度能力 (Independent Thread Scheduling)，可以一定程度上缓解发散带来的影响，" << std::endl;
    //std::cout << "   但理解和最小化发散仍然是重要的优化手段。可以使用 `__syncwarp()` 来确保 warp 在特定点重同步。" << std::endl;
    // 中文注释结束

    // --- 清理 ---
    checkCuda(cudaFree(d_data), "cudaFree d_data");
    checkCuda(cudaEventDestroy(start_event));
    checkCuda(cudaEventDestroy(stop_event));

    // 字符串字面量保持英文
    std::cout << "\nControl flow example finished." << std::endl;
    return 0;
}