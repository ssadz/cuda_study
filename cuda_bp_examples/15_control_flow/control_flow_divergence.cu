#include <iostream>
#include <vector>
#include <numeric>      // For std::iota
#include <algorithm>    // For std::all_of
#include <chrono>
#include <cmath>        // For std::abs

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For warpSize

// 辅助函数：检查 CUDA API 调用结果
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

// 核函数 2: 避免 Warp 发散 (所有线程执行相同路径，通过计算选择值)
// 教学: 通过计算而不是分支来决定操作数，可以减少或避免发散。
//       虽然增加了计算量，但在某些情况下，避免发散的收益更大。
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

    // warpId = threadIdx.x / warpSize; (warpSize 在较新 CUDA 版本中是内置变量)
    // 这里我们假设 blockDim.x 是 warpSize 的整数倍
    int warp_id_in_block = threadIdx.x / warpSize; // warpSize 是内置的

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
    std::cout << "CUDA Control Flow - Branch Divergence Example (Best Practices Chapter 15)" << std::endl;

    const int N = 1024 * 1024 * 8; // 8M elements
    const size_t dataSize = N * sizeof(float);

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
        h_data_expected_coalesced[i] = h_data_expected_divergent[i]; // coalescedKernel 应该得到相同结果

        // warp uniform expected
        int warp_id_in_block_cpu = (i % 256) / 32; // 假设 blockSize 256, warpSize 32
        if ( (i / 256 * (256/32) + warp_id_in_block_cpu) % 2 == 0) { // 模拟全局 warp id
             h_data_expected_warp_uniform[i] = val * 3.0f - 5.0f;
        } else {
             h_data_expected_warp_uniform[i] = val * 0.5f + 5.0f;
        }
    }


    // --- 设备数据 ---
    float* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, dataSize), "cudaMalloc d_data");

    // CUDA 执行配置
    int threadsPerBlock = 256; // 确保是 warpSize (32) 的倍数
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_event, stop_event;
    checkCuda(cudaEventCreate(&start_event));
    checkCuda(cudaEventCreate(&stop_event));
    float ms;

    // --- 1. 测试 Divergent Branch Kernel ---
    std::cout << "\n--- Testing Divergent Branch Kernel ---" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for divergent");

    checkCuda(cudaEventRecord(start_event));
    divergentBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "divergentBranchKernel launch");
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize divergent");
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event));
    std::cout << "Divergent Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_divergent.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for divergent");
    bool divergent_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_data_divergent[i] - h_data_expected_divergent[i]) > 1e-4) { divergent_ok = false; break;}
    std::cout << "Divergent Branch Kernel Verification: " << (divergent_ok ? "PASSED" : "FAILED") << std::endl;


    // --- 2. 测试 Coalesced Branch Kernel ---
    std::cout << "\n--- Testing Coalesced Branch Kernel (emulating branch) ---" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for coalesced");

    checkCuda(cudaEventRecord(start_event));
    coalescedBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "coalescedBranchKernel launch");
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize coalesced");
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event));
    std::cout << "Coalesced Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_coalesced.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for coalesced");
    bool coalesced_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_data_coalesced[i] - h_data_expected_coalesced[i]) > 1e-4) { coalesced_ok = false; break;}
    std::cout << "Coalesced Branch Kernel Verification: " << (coalesced_ok ? "PASSED" : "FAILED") << std::endl;

    // --- 3. 测试 Warp Uniform Branch Kernel ---
    std::cout << "\n--- Testing Warp Uniform Branch Kernel ---" << std::endl;
    // 更新期望结果，因为 warpUniformBranchKernel 的逻辑不同
     for(int i = 0; i < N; ++i) {
        float val = h_data_original[i];
        // 关键在于如何确定线程 i 属于哪个 warp，以及该 warp 的 warp_id_in_block
        // 假设 block 0 的 warp 0, 1, 2... block 1 的 warp 0, 1, 2...
        // (blockId * warps_per_block) + (threadId_in_block / warpSize)
        int current_block_id = i / threadsPerBlock;
        int thread_in_block_id = i % threadsPerBlock;
        int warp_id_in_block_for_cpu = thread_in_block_id / 32; // Assuming warpSize is 32

        if (warp_id_in_block_for_cpu % 2 == 0) { // 模拟核函数逻辑
             h_data_expected_warp_uniform[i] = val * 3.0f - 5.0f;
        } else {
             h_data_expected_warp_uniform[i] = val * 0.5f + 5.0f;
        }
    }
    checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D for warp_uniform");

    checkCuda(cudaEventRecord(start_event));
    warpUniformBranchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaGetLastError(), "warpUniformBranchKernel launch");
    checkCuda(cudaEventRecord(stop_event));
    checkCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize warp_uniform");
    checkCuda(cudaEventElapsedTime(&ms, start_event, stop_event));
    std::cout << "Warp Uniform Branch Kernel Time: " << ms << " ms" << std::endl;

    checkCuda(cudaMemcpy(h_data_warp_uniform.data(), d_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H for warp_uniform");
    bool warp_uniform_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_data_warp_uniform[i] - h_data_expected_warp_uniform[i]) > 1e-4) {
        //std::cerr << "Warp uniform mismatch at " << i << ": Got " << h_data_warp_uniform[i] << " Expected " << h_data_expected_warp_uniform[i] << std::endl;
        warp_uniform_ok = false;
        //break; // Comment out to see more errors if any
    }
    std::cout << "Warp Uniform Branch Kernel Verification: " << (warp_uniform_ok ? "PASSED" : "FAILED") << std::endl;


    std::cout << "\n教学要点 (Best Practices Guide - Chapter 15.1 Branching and Divergence):" << std::endl;
    std::cout << "1. divergentBranchKernel: 当 warp 内的线程根据各自的 `idx % 2` 执行不同代码路径时，会发生分支发散。" << std::endl;
    std::cout << "   这通常会导致性能下降，因为不同的路径需要被硬件串行化执行。" << std::endl;
    std::cout << "2. coalescedBranchKernel: 尝试通过计算（三元操作符）来选择最终值，而不是创建两个截然不同的执行路径。" << std::endl;
    std::cout << "   目标是让所有线程执行相同的指令序列，尽管它们操作的数据可能不同或最终结果不同。" << std::endl;
    std::cout << "   这种方法是否比发散分支更快，取决于具体计算的复杂度和分支的代价。" << std::endl;
    std::cout << "3. warpUniformBranchKernel: 当分支条件对于一个 warp 内的所有线程结果都相同时（例如，条件基于 warp ID），" << std::endl;
    std::cout << "   那么该 warp 内的线程将走同一路径，不会发生 warp 内发散，从而保持较好的性能。" << std::endl;
    std::cout << "4. 实际性能差异取决于 GPU 架构、编译器优化以及分支内代码的复杂度。" << std::endl;
    std::cout << "   通常，应尽量避免或减少不必要的 warp 发散。" << std::endl;
    std::cout << "   如果发散不可避免，尝试使发散的代码段尽可能短。" << std::endl;
    std::cout << "   现代 GPU (如 Volta 及之后架构) 具有独立线程调度能力，可以一定程度上缓解发散带来的影响，" << std::endl;
    std::cout << "   但理解和最小化发散仍然是重要的优化手段。" << std::endl;

    // --- 清理 ---
    checkCuda(cudaFree(d_data), "cudaFree d_data");
    checkCuda(cudaEventDestroy(start_event));
    checkCuda(cudaEventDestroy(stop_event));

    std::cout << "\nControl flow example finished." << std::endl;
    return 0;
}