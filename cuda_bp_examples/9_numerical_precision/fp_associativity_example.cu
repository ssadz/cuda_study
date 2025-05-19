#include <iostream>
#include <vector>
#include <numeric>      // std::iota, std::accumulate
#include <iomanip>      // std::fixed, std::setprecision
#include <cmath>        // std::fabs
#include <algorithm>    // std::generate
#include <random>       // 用于生成随机浮点数

#include <cuda_runtime.h>

// 检查CUDA错误的辅助宏
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            /* 字符串字面量保持英文 */ \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(err_) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 核函数1: 简单的分块并行求和
// 每个线程块计算输入数据的一个分片的和，结果存储在 block_sums 数组中
__global__ void blockSumKernel(const float* input, float* block_sums, int n, int elements_per_block) {
    extern __shared__ float sdata[]; // 动态共享内存

    unsigned int tid_in_block = threadIdx.x;
    unsigned int block_start_idx = blockIdx.x * elements_per_block;
    unsigned int warp_size = warpSize; // CUDA 内置 warp 大小

    // 将数据从全局内存加载到共享内存
    // 每个线程加载多个元素以减少全局内存读取次数并增加计算密度
    float local_sum = 0.0f;
    for (unsigned int i = tid_in_block; i < elements_per_block; i += blockDim.x) {
        if (block_start_idx + i < n) {
            local_sum += input[block_start_idx + i];
        }
    }
    sdata[tid_in_block] = local_sum;
    __syncthreads(); // 确保所有线程都已将其部分和写入共享内存

    // 在共享内存中进行规约 (并行求和)
    // 这是一个常见的并行规约模式，逐步减少活动线程数
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads(); // 同步，确保上一轮的加法完成
    }
    
    // 线程块的第一个线程将最终的部分和写入全局内存
    if (tid_in_block == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}


// 主机端函数，用于计算数组和 (作为参考)
float cpuSum(const std::vector<float>& arr) {
    double sum = 0.0; // 使用 double 以获得更高精度，减少累积误差
    for (float val : arr) {
        sum += val;
    }
    return static_cast<float>(sum);
}

// 主机端函数，使用不同的累加顺序
float cpuSumReverse(const std::vector<float>& arr) {
    double sum = 0.0;
    for (int i = arr.size() - 1; i >= 0; --i) {
        sum += arr[i];
    }
    return static_cast<float>(sum);
}


int main() {
    // 字符串字面量保持英文
    std::cout << "CUDA Floating-Point Non-Associativity Example (Best Practices Chapter 9.3.2)" << std::endl;
    std::cout << std::fixed << std::setprecision(10); // 设置输出精度

    const int N = 1024 * 1024 * 4; // 4M 元素
    const size_t dataSize = N * sizeof(float);

    // --- 主机数据准备 ---
    std::vector<float> h_input(N);

    // 初始化输入数据 (使用一些可能导致舍入误差差异的浮点数)
    std::mt19937 rng(12345); // 固定的随机种子以保证可复现性
    std::uniform_real_distribution<float> dist(0.0001f, 0.001f);
    std::generate(h_input.begin(), h_input.end(), [&]() { return dist(rng); });
    // 添加一些大数和小数混合，更容易观察到差异
    for(int i = 0; i < N / 100; ++i) {
        h_input[i * 100] += static_cast<float>(i % 2 == 0 ? 100.0f : -100.0f);
    }


    // --- CPU 参考求和 ---
    // 字符串字面量保持英文
    std::cout << "\n--- Calculating sums on CPU ---" << std::endl;
    float sum_cpu_forward = cpuSum(h_input);
    float sum_cpu_reverse = cpuSumReverse(h_input);
    // 字符串字面量保持英文
    std::cout << "CPU Sum (forward order):    " << sum_cpu_forward << std::endl;
    std::cout << "CPU Sum (reverse order):    " << sum_cpu_reverse << std::endl;
    std::cout << "Difference (CPU forward vs reverse): " << std::abs(sum_cpu_forward - sum_cpu_reverse) << std::endl;


    // --- GPU 并行求和 ---
    // 字符串字面量保持英文
    std::cout << "\n--- Calculating sum on GPU (Block Sum + CPU final sum) ---" << std::endl;

    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, dataSize));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), dataSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    // elements_per_block 可以根据需要调整，这里简单设为与 threadsPerBlock 相同，即每个线程初始处理一个元素
    // 实际上在 blockSumKernel 内部，每个线程会迭代处理多个元素以覆盖整个 elements_per_block
    int elements_per_block_gpu = 2048; // 每个块处理的元素数量
    if (N < elements_per_block_gpu) elements_per_block_gpu = N;

    int numBlocks = (N + elements_per_block_gpu - 1) / elements_per_block_gpu;
    
    std::vector<float> h_block_sums(numBlocks);
    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    // 动态共享内存大小：每个线程块需要 threadsPerBlock 个浮点数的空间
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    float ms;

    CUDA_CHECK(cudaEventRecord(start_event));
    blockSumKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_block_sums, N, elements_per_block_gpu);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "GPU blockSumKernel Time: " << ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // 在 CPU 上对块的部分和进行最终求和
    float sum_gpu_block_reduce = 0.0f;
    double sum_gpu_block_reduce_double = 0.0; // 使用 double 累加
    for (float val : h_block_sums) {
        sum_gpu_block_reduce_double += val;
    }
    sum_gpu_block_reduce = static_cast<float>(sum_gpu_block_reduce_double);

    // 字符串字面量保持英文
    std::cout << "GPU Sum (block reduce then CPU sum): " << sum_gpu_block_reduce << std::endl;
    std::cout << "Difference (CPU forward vs GPU): " << std::abs(sum_cpu_forward - sum_gpu_block_reduce) << std::endl;


    // --- 教学要点 ---
    // 字符串字面量保持英文 (std::cout 部分)
    std::cout << "\nTeaching Points (Best Practices Guide - Chapter 9.3.2 Floating Point Math Is Not Associative):" << std::endl;
    // 中文注释开始 (对下面英文教学点的解释)
    // 1. 浮点数运算（尤其是加法）在计算机中不完全满足结合律 `(a+b)+c == a+(b+c)`，因为每次运算都可能引入舍入误差。
    // 2. 不同的求和顺序（例如 CPU 顺序求和、CPU 逆序求和、GPU 并行分块求和再汇总）可能会导致最终结果有微小差异。
    // 3. 在本例中，CPU 顺序累加、CPU 逆序累加和 GPU 并行累加（其内部也有特定的累加顺序）的结果可能都不同。
    // 4. 这种差异是浮点运算的固有特性，在并行化算法时尤其需要注意，特别是对于归约操作。
    // 5. 在验证并行算法的正确性时，不应期望与串行版本结果完全位对位相同，而应设置一个可接受的误差容限 (epsilon)。
    // 6. 对于需要高精度的求和，可以在中间步骤或最终汇总时使用更高精度的数据类型（如 `double`），如本例中 CPU 求和和 GPU 块结果汇总时所示。
    // 7. CUDA 提供了如 `cub::DeviceReduce` (来自 CUB 库) 这样的高度优化的并行归约 primitives，它们在性能和数值稳定性方面通常做得很好。
    //    对于更复杂的归约，建议使用这些库函数。
    // 中文注释结束
    std::cout << "1. Floating-point operations (especially addition) are not strictly associative `(a+b)+c == a+(b+c)` in computers due to rounding errors in each operation." << std::endl;
    std::cout << "2. Different summation orders (e.g., CPU forward sum, CPU reverse sum, GPU parallel block sum then final sum) can lead to slightly different final results." << std::endl;
    std::cout << "3. In this example, the sum from CPU forward, CPU reverse, and GPU (which also has its own internal summation order) might all differ slightly." << std::endl;
    std::cout << "4. This discrepancy is an inherent characteristic of floating-point arithmetic and is particularly important to consider when parallelizing algorithms, especially reductions." << std::endl;
    std::cout << "5. When verifying parallel algorithms, results should often be compared within an acceptable tolerance (epsilon) rather than expecting bitwise equality with serial versions." << std::endl;
    std::cout << "6. For sums requiring higher precision, using a higher-precision type (like `double`) for intermediate or final accumulations can help, as shown in CPU sum and GPU block sum aggregation." << std::endl;
    std::cout << "7. CUDA libraries like CUB (`cub::DeviceReduce`) provide highly optimized parallel reduction primitives that are generally good for performance and numerical stability." << std::endl;


    // --- 清理 ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    // 字符串字面量保持英文
    std::cout << "\nFloating-point associativity example finished." << std::endl;
    return 0;
}