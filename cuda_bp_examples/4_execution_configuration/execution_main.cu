#include <iostream>
#include <vector>
#include <numeric>   // 用于 std::iota (for std::iota)
#include <algorithm> // 用于 std::fill, std::all_of (for std::fill, std::all_of)
#include <chrono>    // 用于计时 (for timing)

// CUDA 运行时 API (CUDA Runtime API)
#include <cuda_runtime.h>

// 辅助函数: 检查 CUDA API 调用结果
// (Helper function: Check CUDA API call result)
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        // CUDA 运行时错误:
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    }
    return result;
}

// 简单的核函数: 将数组中的每个元素加上一个值
// (Simple kernel: Add a value to each element of an array)
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) 是一个可选的启动边界限定符.
// (- maxThreadsPerBlock: Hints to the compiler that the kernel will not be launched with more threads per block than this value.)
// (  The compiler can use this information to optimize resource usage (e.g., register allocation).)
// (  If omitted, the compiler assumes a conservative value.)
// (- minBlocksPerMultiprocessor: (Optional) Hints to the compiler how many blocks can reside on an SM at a minimum.)
// (  This can help the compiler balance resource usage to improve occupancy.)
// 选择合适的 maxThreadsPerBlock 值:
// (Choosing an appropriate maxThreadsPerBlock value:)
//   - 必须是编译时常量. (Must be a compile-time constant.)
//   - 通常设置为核函数设计时预期的最大或典型块大小. (Usually set to the maximum or typical block size expected during kernel design.)
//   - 如果设置得太低, 而实际启动配置使用了更大的块, 编译器可能无法充分优化,
//     或者在某些情况下 (如果寄存器溢出到本地内存), 可能会导致性能下降.
//     (If set too low, and the actual launch configuration uses larger blocks, the compiler might not optimize sufficiently,
//      or in some cases (if registers spill to local memory), it might lead to performance degradation.)
//   - 如果设置得太高, 或者不设置, 编译器可能会为更高的线程数优化,
//     当使用较少线程数启动时, 这可能不是最优的.
//     (If set too high, or not set, the compiler might optimize for a higher thread count,
//      which might not be optimal when launching with fewer threads.)
// 这里的 1024 是一个常见的最大值, 因为许多设备每个块最多支持 1024 个线程.
// (1024 here is a common maximum, as many devices support up to 1024 threads per block.)
__global__ void addValueToElements(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += value;
    }
}

// 不使用 __launch_bounds__ 的版本, 用于对比
// (Version without __launch_bounds__, for comparison)
__global__ void addValueToElements_no_bounds(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += value;
    }
}

// 核函数: 简单地将线程ID写入输出数组, 用于观察线程如何映射
// (Kernel: Simply writes the thread ID to an output array, for observing thread mapping)
__global__ void getThreadGlobalIndex(int* output_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output_data[idx] = idx;
    }
}

int main() {
    // CUDA 执行配置优化指南 (最佳实践第13章)
    std::cout << "CUDA Execution Configuration Optimization Tutorial (Chapter 13 Best Practices)" << std::endl;

    const int N = 1024 * 1024; // 数组大小 (1M 元素) (Array size (1M elements))
    const size_t dataSize = N * sizeof(float);
    const size_t intDataSize = N * sizeof(int);

    // --- 主机数据 --- (Host data)
    std::vector<float> h_data(N);
    std::vector<float> h_data_expected(N);
    std::vector<int> h_idx_data_output(N); // 用于 getThreadGlobalIndex (for getThreadGlobalIndex)

    // 初始化数据 (Initialize data)
    std::iota(h_data.begin(), h_data.end(), 1.0f); // 1.0f, 2.0f, ...
    for(int i = 0; i < N; ++i) {
        h_data_expected[i] = h_data[i] + 10.0f; // 预期结果 (Expected result)
    }

    // --- 设备数据 --- (Device data)
    float* d_data = nullptr;
    int* d_idx_data = nullptr;

    // 分配设备内存 (Allocate device memory)
    if (checkCuda(cudaMalloc((void**)&d_data, dataSize)) != cudaSuccess) return 1;
    if (checkCuda(cudaMalloc((void**)&d_idx_data, intDataSize)) != cudaSuccess) {
        checkCuda(cudaFree(d_data));
        return 1;
    }

    // 理解执行配置 (网格和块维度)
    std::cout << "\n--- 1. Understanding Execution Configuration (Grid and Block Dimensions) ---" << std::endl;
    // CUDA 核函数以一个网格 (Grid) 的线程块 (Blocks) 来启动.
    // (CUDA kernels are launched as a grid of thread blocks.)
    // 每个线程块包含一定数量的线程 (Threads).
    // (Each thread block contains a certain number of threads.)
    //
    // - 线程 (Thread): CUDA 中最基本的执行单元. (Thread: The most basic execution unit in CUDA.)
    // - 线程块 (Block): 一组协同执行的线程. (Block: A group of threads executing cooperatively.)
    //   - 同一个块内的线程可以通过共享内存 (Shared Memory) 进行通信和同步 (__syncthreads()).
    //     (Threads within the same block can communicate and synchronize via Shared Memory (__syncthreads()).)
    //   - 块内的线程在同一个流式多处理器 (SM) 上执行.
    //     (Threads in a block execute on the same Streaming Multiprocessor (SM).)
    //   - 块可以组织成1D, 2D, 或 3D. 通过 blockDim.x, blockDim.y, blockDim.z 访问.
    //     (Blocks can be organized in 1D, 2D, or 3D. Accessed via blockDim.x, blockDim.y, blockDim.z.)
    // - 网格 (Grid): 一组线程块. (Grid: A group of thread blocks.)
    //   - 不同块之间的线程不能直接通信或同步 (除非使用非常高级的技术, 如动态并行或全局同步).
    //     (Threads in different blocks cannot directly communicate or synchronize (except with advanced techniques like dynamic parallelism or global synchronization).)
    //   - 网格可以组织成1D, 2D, 或 3D. 通过 gridDim.x, gridDim.y, gridDim.z 访问.
    //     (Grids can be organized in 1D, 2D, or 3D. Accessed via gridDim.x, gridDim.y, gridDim.z.)
    //
    // 线程的全局唯一 ID 计算:
    // (Global unique ID calculation for a thread:)
    // 对于 1D Grid 和 1D Block:
    // (For 1D Grid and 1D Block:)
    //   int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //
    // 选择执行配置的目标:
    // (Goals for choosing execution configuration:)
    // 1. 保证正确性: 启动足够的线程来处理所有数据元素.
    //    (Ensure correctness: Launch enough threads to process all data elements.)
    // 2. 提高性能: 最大化 GPU 的利用率 (占有率 Occupancy) 并有效隐藏延迟.
    //    (Improve performance: Maximize GPU utilization (Occupancy) and effectively hide latency.)
    //
    // 占有率 (Occupancy):
    //   指的是 SM 上活跃的 Warp (线程束, 通常32个线程) 数量与 SM 最大支持的活跃 Warp 数量的比例.
    //   (Refers to the ratio of active Warps (a group of 32 threads) on an SM to the maximum number of active Warps supported by the SM.)
    //   更高的占有率通常有助于隐藏内存访问和指令执行的延迟, 因为 SM 可以在一个 Warp 等待时切换到另一个活跃的 Warp.
    //   (Higher occupancy usually helps hide memory access and instruction execution latency, as the SM can switch to another active Warp while one is waiting.)
    //   但是, 并非占有率越高越好. 过高的占有率可能意味着每个线程可用的资源 (如寄存器、共享内存) 减少,
    //   可能导致寄存器溢出到本地内存 (速度慢) 或共享内存不足.
    //   (However, higher occupancy is not always better. Too high occupancy might mean fewer resources (like registers, shared memory) available per thread,
    //    potentially leading to register spilling to local memory (which is slow) or insufficient shared memory.)
    //
    // 影响占有率的因素:
    // (Factors affecting occupancy:)
    //   - 每块线程数 (blockDim): 块越大, 需要的 SM 资源越多 (共享内存、寄存器总数).
    //     (Threads per block (blockDim): Larger blocks require more SM resources (shared memory, total registers).)
    //   - 每个线程使用的寄存器数: 由编译器决定, 受代码复杂度和 __launch_bounds__ 影响.
    //     (Registers used per thread: Determined by the compiler, influenced by code complexity and __launch_bounds__.)
    //   - 每个块使用的共享内存量: 由核函数定义.
    //     (Amount of shared memory used per block: Defined by the kernel.)
    //   - SM 的硬件限制 (最大线程数、最大块数、总寄存器数、总共享内存).
    //     (SM hardware limits (max threads, max blocks, total registers, total shared memory).)
    //
    // 通常的块大小选择:
    // (Typical block size selection:)
    //   - 每块线程数应为 Warp 大小的整数倍 (通常是 32 的倍数, 如 64, 128, 256, 512, 1024).
    //     (Threads per block should be a multiple of the Warp size (usually a multiple of 32, e.g., 64, 128, 256, 512, 1024).)
    //     这有助于确保 Warp 被充分利用, 并简化某些内存访问模式的分析.
    //     (This helps ensure Warps are fully utilized and simplifies the analysis of certain memory access patterns.)
    //   - 推荐范围通常在 128 到 256 个线程/块之间作为起点, 但最佳值取决于具体核函数和 GPU 架构.
    //     (The recommended range is typically between 128 and 256 threads/block as a starting point, but the optimal value depends on the specific kernel and GPU architecture.)
    //   - NVIDIA 提供了 CUDA Occupancy Calculator 工具来帮助分析和选择.
    //     (NVIDIA provides the CUDA Occupancy Calculator tool to help with analysis and selection.)

    // 示例: 启动 getThreadGlobalIndex 核函数并观察其输出
    // (Example: Launch the getThreadGlobalIndex kernel and observe its output)
    // 我们将使用不同的块大小, 但保持总线程数至少为 N
    // (We will use different block sizes, but keep the total number of threads at least N)
    std::cout << "  Testing getThreadGlobalIndex kernel..." << std::endl;
    int test_N = 32; // 用一个小 N 来打印和检查索引 (Use a small N to print and check indices)
    if (checkCuda(cudaMemset(d_idx_data, 0, test_N * sizeof(int))) != cudaSuccess) return 1; // 清零 (Zero out)

    int threadsPerBlock_idx_test = 8; // 每块8个线程 (8 threads per block)
    int blocksPerGrid_idx_test = (test_N + threadsPerBlock_idx_test - 1) / threadsPerBlock_idx_test; // 保证覆盖 test_N (Ensure coverage for test_N)

    std::cout << "    Launch config: " << blocksPerGrid_idx_test << " blocks, "
              << threadsPerBlock_idx_test << " threads/block" << std::endl;

    getThreadGlobalIndex<<<blocksPerGrid_idx_test, threadsPerBlock_idx_test>>>(d_idx_data, test_N);
    checkCuda(cudaGetLastError()); // 检查核函数启动错误 (Check for kernel launch errors)
    checkCuda(cudaDeviceSynchronize()); // 等待核函数完成 (Wait for kernel to complete)

    checkCuda(cudaMemcpy(h_idx_data_output.data(), d_idx_data, test_N * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "    Global thread indices (first " << test_N << " values): ";
    for (int i = 0; i < test_N; ++i) {
        std::cout << h_idx_data_output[i] << (i == test_N - 1 ? "" : ", ");
    }
    std::cout << std::endl;
    // 预期输出应该是 0, 1, 2, ..., 31 (Expected output should be 0, 1, 2, ..., 31)

    // 不同执行配置的性能影响 (概念性)
    std::cout << "\n--- 2. Performance Impact of Different Execution Configurations (Conceptual) ---" << std::endl;
    // 为了演示, 我们将使用 addValueToElements 核函数, 并尝试几种不同的块大小.
    // (For demonstration, we will use the addValueToElements kernel and try several different block sizes.)
    // 注意: 这个简单核函数的性能差异可能不明显, 因为其计算量很小.
    // (Note: The performance difference for this simple kernel might not be significant because its computational load is small.)
    // 在实际应用中, 更复杂的核函数对执行配置会更敏感.
    // (In real applications, more complex kernels are more sensitive to execution configuration.)

    float value_to_add = 10.0f;
    int iterations = 5; // 多次运行取平均或观察稳定性 (简单演示中不详细计时)
                        // (Run multiple times to average or observe stability (timing not detailed in this simple demo))

    // 配置 1: 较小的块大小 (Configuration 1: Smaller block size)
    int t_config1 = 64; // 64 线程/块 (64 threads/block)
    int b_config1 = (N + t_config1 - 1) / t_config1;
    std::cout << "  Test Config 1: " << b_config1 << " blocks, " << t_config1 << " threads/block" << std::endl;
    
    checkCuda(cudaMemcpy(d_data, h_data.data(), dataSize, cudaMemcpyHostToDevice)); // 重置设备数据 (Reset device data)
    auto start_c1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        addValueToElements_no_bounds<<<b_config1, t_config1>>>(d_data, N, value_to_add);
    }
    checkCuda(cudaDeviceSynchronize()); // 确保所有迭代完成 (Ensure all iterations are complete)
    auto end_c1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_c1 = (end_c1 - start_c1) / iterations;
    std::cout << "    Config 1 Avg Time: " << duration_c1.count() << " ms" << std::endl;
    // (验证结果的代码可以加在这里, 但为了简洁暂时省略, 假设核函数正确)
    // (Verification code could be added here, but omitted for brevity, assuming the kernel is correct)

    // 配置 2: 中等块大小 (通常是较好的起点)
    // (Configuration 2: Medium block size (often a good starting point))
    int t_config2 = 256; // 256 线程/块 (256 threads/block)
    int b_config2 = (N + t_config2 - 1) / t_config2;
    std::cout << "  Test Config 2: " << b_config2 << " blocks, " << t_config2 << " threads/block" << std::endl;

    checkCuda(cudaMemcpy(d_data, h_data.data(), dataSize, cudaMemcpyHostToDevice));
    auto start_c2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        addValueToElements_no_bounds<<<b_config2, t_config2>>>(d_data, N, value_to_add);
    }
    checkCuda(cudaDeviceSynchronize());
    auto end_c2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_c2 = (end_c2 - start_c2) / iterations;
    std::cout << "    Config 2 Avg Time: " << duration_c2.count() << " ms" << std::endl;

    // 配置 3: 较大的块大小 (通常是硬件支持的上限附近)
    // (Configuration 3: Larger block size (often near the hardware support limit))
    int t_config3 = 1024; // 1024 线程/块 (1024 threads/block)
    int b_config3 = (N + t_config3 - 1) / t_config3;
    std::cout << "  Test Config 3: " << b_config3 << " blocks, " << t_config3 << " threads/block" << std::endl;
    
    checkCuda(cudaMemcpy(d_data, h_data.data(), dataSize, cudaMemcpyHostToDevice));
    auto start_c3 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        addValueToElements_no_bounds<<<b_config3, t_config3>>>(d_data, N, value_to_add);
    }
    checkCuda(cudaDeviceSynchronize());
    auto end_c3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_c3 = (end_c3 - start_c3) / iterations;
    std::cout << "    Config 3 Avg Time: " << duration_c3.count() << " ms" << std::endl;
    
    // 从设备复制最终结果并验证 (使用最后一次运行的配置)
    // (Copy final result from device and verify (using the last run's configuration))
    std::vector<float> h_result_c3(N);
    checkCuda(cudaMemcpy(h_result_c3.data(), d_data, dataSize, cudaMemcpyDeviceToHost));
    bool all_correct_c3 = true;
    float expected_value_c3;
    for(int i=0; i < N; ++i) {
        expected_value_c3 = h_data[i] + iterations * value_to_add; // 修正预期值 (Corrected expected value)
        if (std::abs(h_result_c3[i] - expected_value_c3) > 1e-5) {
            all_correct_c3 = false;
            // 配置 3 结果错误! 索引 ... 预期 ... 得到 ...
            std::cerr << "  Config 3 Result Error! Index " << i << ", Expected " << expected_value_c3 << ", Got " << h_result_c3[i] << std::endl;
            break;
        }
    }
    if(all_correct_c3) {
        // 配置 3 结果正确!
         std::cout << "  Config 3 Result OK!" << std::endl;
    } else {
        // 配置 3 结果失败!
         std::cout << "  Config 3 Result FAIL!" << std::endl;
    }

    // 使用 __launch_bounds__
    std::cout << "\n--- 3. Using __launch_bounds__ ---" << std::endl;
    // __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
    // - MAX_THREADS_PER_BLOCK: 告知编译器此核函数启动时, 每块线程数不会超过这个值.
    //   (Tells the compiler that when this kernel is launched, the number of threads per block will not exceed this value.)
    //   编译器可以利用这个信息优化寄存器分配. 如果启动时超过此值, 行为未定义 (通常编译时报错或运行时错误).
    //   (The compiler can use this information to optimize register allocation. If launched with more than this value, the behavior is undefined (usually a compile-time or runtime error).)
    // - MIN_BLOCKS_PER_SM (可选): 期望每个 SM 至少能同时运行多少个这样的块.
    //   ((Optional) The desired minimum number of such blocks that can run concurrently on each SM.)
    //   这有助于编译器调整资源 (主要是寄存器) 使用, 以尝试达到目标占有率.
    //   (This helps the compiler adjust resource (mainly register) usage to try to achieve the target occupancy.)
    //
    // 好处: (Benefits:)
    //   - 寄存器优化: 明确告知编译器线程数的上限, 编译器可能分配更少的寄存器给每个线程,
    //     从而允许更多线程块并发执行在 SM 上, 提高占有率.
    //     (Register optimization: Explicitly informs the compiler of the upper limit on thread count, allowing the compiler to potentially allocate fewer registers per thread,
    //      thus enabling more thread blocks to execute concurrently on an SM, increasing occupancy.)
    //   - 提高可移植性和可维护性: 代码更清晰地表达了对执行配置的预期.
    //     (Improved portability and maintainability: The code more clearly expresses expectations for the execution configuration.)
    //
    // 何时使用: (When to use:)
    //   - 当你知道核函数大部分情况下会以某个特定的最大块大小启动时.
    //     (When you know the kernel will mostly be launched with a specific maximum block size.)
    //   - 当你需要更精细地控制寄存器使用以提高占有率时.
    //     (When you need finer control over register usage to improve occupancy.)
    //
    // 注意: addValueToElements 核函数已经定义了 __launch_bounds__(1024).
    // (Note: The addValueToElements kernel has already defined __launch_bounds__(1024).)
    // 我们将用它和 addValueToElements_no_bounds (未定义launch_bounds) 进行概念性对比.
    // (We will compare it conceptually with addValueToElements_no_bounds (which has no launch_bounds defined).)
    // 实际性能差异高度依赖于核函数的复杂度和目标硬件. 对于这个简单核函数, 差异可能微乎其微.
    // (The actual performance difference is highly dependent on the kernel's complexity and the target hardware. For this simple kernel, the difference might be negligible.)

    // 测试带 __launch_bounds__(1024) 的核函数 (使用配置 2 参数)
    std::cout << "  Testing kernel with __launch_bounds__(1024) (using Config 2 params):" << std::endl;
    checkCuda(cudaMemcpy(d_data, h_data.data(), dataSize, cudaMemcpyHostToDevice));
    auto start_lb = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        addValueToElements<<<b_config2, t_config2>>>(d_data, N, value_to_add); // 使用带 bounds 的核函数 (Use kernel with bounds)
    }
    checkCuda(cudaDeviceSynchronize());
    auto end_lb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_lb = (end_lb - start_lb) / iterations;
    // 带 __launch_bounds__ 平均时间
    std::cout << "    With __launch_bounds__ Avg Time: " << duration_lb.count() << " ms" << std::endl;

    // 验证结果 (Verify results)
    std::vector<float> h_result_lb(N);
    checkCuda(cudaMemcpy(h_result_lb.data(), d_data, dataSize, cudaMemcpyDeviceToHost));
    bool all_correct_lb = true;
    float expected_value_lb;
    for(int i=0; i < N; ++i) {
        expected_value_lb = h_data[i] + iterations * value_to_add; // 修正预期值 (Corrected expected value)
        if (std::abs(h_result_lb[i] - expected_value_lb) > 1e-5) {
            all_correct_lb = false;
            // 带 __launch_bounds__ 结果错误! 索引 ... 预期 ... 得到 ...
            std::cerr << "  With __launch_bounds__ Result Error! Index " << i << ", Expected " << expected_value_lb << ", Got " << h_result_lb[i] << std::endl;
            break;
        }
    }
    if(all_correct_lb) {
        // 带 __launch_bounds__ 结果正确!
         std::cout << "  With __launch_bounds__ Result OK!" << std::endl;
    } else {
        // 带 __launch_bounds__ 结果失败!
         std::cout << "  With __launch_bounds__ Result FAIL!" << std::endl;
    }

    // 总结
    std::cout << "\n--- Summary ---" << std::endl;
    // 1. 推荐的块大小范围: 64-1024 线程/块, 通常 256 是良好的起点.
    std::cout << "  1. Recommended block size range: 64-1024 threads/block, 256 is often a good starting point." << std::endl;
    // 2. 网格大小计算公式: (元素数量 + 块大小 - 1) / 块大小.
    std::cout << "  2. Grid size calculation formula: (number_of_elements + block_size - 1) / block_size." << std::endl;
    // 3. __launch_bounds__ 可用于优化寄存器使用.
    std::cout << "  3. __launch_bounds__ can be used to optimize register usage." << std::endl;
    // 4. CUDA Occupancy Calculator 工具可帮助分析占有率.
    std::cout << "  4. The CUDA Occupancy Calculator tool can help analyze occupancy." << std::endl;

    // --- 清理 --- (Cleanup)
    // 清理设备内存
    std::cout << "\n--- Cleanup Device Memory ---" << std::endl;
    checkCuda(cudaFree(d_data));
    checkCuda(cudaFree(d_idx_data));
    // 设备内存已释放.
    std::cout << "Device memory released." << std::endl;

    // 执行配置优化示例已完成.
    std::cout << "\nExecution configuration optimization example completed." << std::endl;
    return 0;
}
