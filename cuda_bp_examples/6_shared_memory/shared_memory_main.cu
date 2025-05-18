#include <iostream>
#include <vector>
#include <numeric>      // 用于 std::iota
#include <algorithm>    // 用于 std::fill, std::equal
#include <cmath>        // 用于 std::abs
#include <chrono>       // 用于计时

// CUDA 运行时 API
#include <cuda_runtime.h>
// cuBLAS API
#include <cublas_v2.h>

// 辅助函数：检查 CUDA API 调用结果
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    }
    return result;
}

// 辅助函数：检查 cuBLAS API 调用结果
inline cublasStatus_t checkCublas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        // cuBLAS 没有像 cudaGetErrorString 这样的直接函数，这里仅打印状态码
        // 在实际项目中，可以维护一个从状态码到错误字符串的映射
        std::cerr << "cuBLAS API Error: Status " << result << std::endl;
    }
    return result;
}

// CPU 版本的矩阵转置
void matrixTransposeCPU(const float* input, float* output, int n) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            output[j * n + i] = input[i * n + j];
        }
    }
}

// CUDA 核函数: 矩阵转置 - 仅使用全局内存 (全局访问不合并)
__global__ void matrixTransposeGlobalMem(const float* input, float* output, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        // 确保索引计算正确，避免溢出
        output[j * n + i] = input[i * n + j];
    }
}

// CUDA 核函数: 矩阵转置 - 使用共享内存优化
#define TILE_DIM 32
#define BLOCK_ROWS 8

// 矩阵转置核函数 - 使用完全消除银行冲突的优化设计
__global__ void matrixTransposeSharedMem(const float* input, float* output, int n) {
    // 关键设计：通过修改填充消除银行冲突
    // 通常GPU有32个内存银行，每个银行宽度为4字节(float)
    // 使用 TILE_DIM+1 的填充可以错开相邻元素的银行访问
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    // 计算全局和局部索引
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 分块加载，提高合并访问效率
    #pragma unroll 4
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < n && x < n) {
            // 将一列数据加载到共享内存的一行
            // 这里采用合并访问，非常高效
            tile[threadIdx.y + j][threadIdx.x] = __ldg(&input[(y + j) * n + x]);
        }
    }
    
    // 等待所有线程完成加载
    __syncthreads();
    
    // 交换块索引，转为转置坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 从共享内存读取并写回全局内存
    #pragma unroll 4
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y + j < n && x < n) {
            // 通过列主序读取共享内存中的数据实现转置
            // 因为填充了+1，不同线程访问不同内存银行，避免了冲突
            output[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    std::cout << "CUDA Matrix Transpose Comparison: Global, Shared Mem, and cuBLAS" << std::endl;

    const int N = 8192; // 矩阵大小 (8192x8192)
    const size_t dataSize = N * N * sizeof(float);

    // --- 主机数据准备 ---
    std::vector<float> h_input(N * N);
    std::vector<float> h_output_cpu(N * N);
    std::vector<float> h_output_gpu_global(N * N);
    std::vector<float> h_output_gpu_shared(N * N);
    std::vector<float> h_output_gpu_cublas(N * N); // 为cuBLAS版本添加

    // 初始化输入数据
    std::iota(h_input.begin(), h_input.end(), 0.0f); // 0, 1, 2, ...

    // --- CPU 计算 (用于验证) ---
    std::cout << "Executing CPU matrix transpose..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixTransposeCPU(h_input.data(), h_output_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution finished. Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- GPU 数据准备 ---
    float* d_input = nullptr;
    float* d_output_global = nullptr;
    float* d_output_shared = nullptr;
    float* d_output_cublas = nullptr; // 为cuBLAS版本添加

    // 分配设备内存
    std::cout << "Allocating GPU memory..." << std::endl;
    if (checkCuda(cudaMalloc((void**)&d_input, dataSize)) != cudaSuccess) return 1;
    if (checkCuda(cudaMalloc((void**)&d_output_global, dataSize)) != cudaSuccess) { cudaFree(d_input); return 1;}
    if (checkCuda(cudaMalloc((void**)&d_output_shared, dataSize)) != cudaSuccess) { cudaFree(d_input); cudaFree(d_output_global); return 1;}
    if (checkCuda(cudaMalloc((void**)&d_output_cublas, dataSize)) != cudaSuccess) { cudaFree(d_input); cudaFree(d_output_global); cudaFree(d_output_shared); return 1;} // 为cuBLAS版本添加
    std::cout << "GPU memory allocation complete." << std::endl;

    // 将输入数据从主机复制到设备
    std::cout << "Copying input data from host to device..." << std::endl;
    if (checkCuda(cudaMemcpy(d_input, h_input.data(), dataSize, cudaMemcpyHostToDevice)) != cudaSuccess) { /* 清理 */ return 1;}
    std::cout << "Input data copying complete." << std::endl;

    // CUDA 执行配置
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS); // TILE_DIM=32, BLOCK_ROWS=8

    // 为共享内存版本计算网格维度
    dim3 dimGridShared((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // 为全局内存版本计算网格维度
    // j (列)方向由 TILE_DIM (blockDim.x) 决定
    // i (行)方向由 BLOCK_ROWS (blockDim.y) 决定
    dim3 dimGridGlobal((N + TILE_DIM - 1) / TILE_DIM, (N + BLOCK_ROWS - 1) / BLOCK_ROWS);

    // --- 1. GPU 执行: 全局内存版本 ---
    std::cout << "Executing GPU matrix transpose (Global Memory version)..." << std::endl;
    cudaEvent_t start_gpu_global, stop_gpu_global;
    checkCuda(cudaEventCreate(&start_gpu_global));
    checkCuda(cudaEventCreate(&stop_gpu_global));

    // 预热部分改为多次运行
    for (int i = 0; i < 5; i++) {
        matrixTransposeGlobalMem<<<dimGridGlobal, dimBlock>>>(d_input, d_output_global, N);
    }
    checkCuda(cudaDeviceSynchronize());

    // 计时开始
    checkCuda(cudaEventRecord(start_gpu_global));
    
    // 运行核函数
    matrixTransposeGlobalMem<<<dimGridGlobal, dimBlock>>>(d_input, d_output_global, N);
    checkCuda(cudaGetLastError()); // 检查核函数启动错误
    
    // 计时结束
    checkCuda(cudaEventRecord(stop_gpu_global));
    checkCuda(cudaEventSynchronize(stop_gpu_global)); // 等待核函数完成
    
    float ms_global_mem = 0;
    checkCuda(cudaEventElapsedTime(&ms_global_mem, start_gpu_global, stop_gpu_global));
    std::cout << "GPU (Global Memory) execution finished. Time: " << ms_global_mem << " ms" << std::endl;
    checkCuda(cudaEventDestroy(start_gpu_global));
    checkCuda(cudaEventDestroy(stop_gpu_global));

    // 将结果从设备复制回主机
    checkCuda(cudaMemcpy(h_output_gpu_global.data(), d_output_global, dataSize, cudaMemcpyDeviceToHost));

    // --- 2. GPU 执行: 共享内存版本 ---
    std::cout << "Executing GPU matrix transpose (Shared Memory version)..." << std::endl;
    
    cudaEvent_t start_gpu_shared, stop_gpu_shared;
    checkCuda(cudaEventCreate(&start_gpu_shared));
    checkCuda(cudaEventCreate(&stop_gpu_shared));
    
    // 预热部分改为多次运行
    for (int i = 0; i < 5; i++) {
        matrixTransposeSharedMem<<<dimGridShared, dimBlock>>>(d_input, d_output_shared, N);
    }
    checkCuda(cudaDeviceSynchronize());
    
    // 计时开始
    checkCuda(cudaEventRecord(start_gpu_shared));
    
    // 运行核函数
    matrixTransposeSharedMem<<<dimGridShared, dimBlock>>>(d_input, d_output_shared, N);
    checkCuda(cudaGetLastError()); // 检查核函数启动错误
    
    // 计时结束
    checkCuda(cudaEventRecord(stop_gpu_shared));
    checkCuda(cudaEventSynchronize(stop_gpu_shared)); // 等待核函数完成

    float ms_shared_mem = 0;
    checkCuda(cudaEventElapsedTime(&ms_shared_mem, start_gpu_shared, stop_gpu_shared));
    float speedup = ms_global_mem / ms_shared_mem;
    std::cout << "GPU (Shared Memory) execution finished. Time: " << ms_shared_mem << " ms" << std::endl;
    std::cout << "Speedup of Shared Memory over Global Memory: " << speedup << "x" << std::endl;
    checkCuda(cudaEventDestroy(start_gpu_shared));
    checkCuda(cudaEventDestroy(stop_gpu_shared));

    // 将结果从设备复制回主机
    checkCuda(cudaMemcpy(h_output_gpu_shared.data(), d_output_shared, dataSize, cudaMemcpyDeviceToHost));

    // --- 3. GPU 执行: cuBLAS 版本 ---
    std::cout << "Executing GPU matrix transpose (cuBLAS version)..." << std::endl;
    cublasHandle_t cublas_handle;
    if (checkCublas(cublasCreate(&cublas_handle)) != CUBLAS_STATUS_SUCCESS) {
        // 清理已分配的内存
        cudaFree(d_input);
        cudaFree(d_output_global);
        cudaFree(d_output_shared);
        cudaFree(d_output_cublas);
        return 1;
    }

    cudaEvent_t start_gpu_cublas, stop_gpu_cublas;
    checkCuda(cudaEventCreate(&start_gpu_cublas));
    checkCuda(cudaEventCreate(&stop_gpu_cublas));
    
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f; // 为 cublasSgeam 添加 beta

    // 预热 (与其它版本保持一致)
    for (int i = 0; i < 5; ++i) {
        // checkCublas(cublasSomatcopy(cublas_handle, CUBLAS_OP_T, N, N, &alpha_one, d_input, N, d_output_cublas, N));
        checkCublas(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, &alpha_one, d_input, N, &beta_zero, d_input, N, d_output_cublas, N));
    }
    checkCuda(cudaDeviceSynchronize()); // 确保预热完成

    // 计时开始
    checkCuda(cudaEventRecord(start_gpu_cublas));
    
    // 执行cuBLAS转置
    // C = A^T,  op(A) is A^T.
    // cublasSomatcopy(handle, trans, rows_of_opA, cols_of_opA, alpha, A, lda, B, ldb)
    // rows_of_opA = N (original_cols_of_A)
    // cols_of_opA = N (original_rows_of_A)
    // lda = N (leading dimension of A)
    // ldb = N (leading dimension of B, B has rows_of_opA rows)
    // checkCublas(cublasSomatcopy(cublas_handle, CUBLAS_OP_T, N, N, &alpha_one, d_input, N, d_output_cublas, N));
    checkCublas(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, &alpha_one, d_input, N, &beta_zero, d_input, N, d_output_cublas, N));
    checkCuda(cudaGetLastError()); // 检查cuBLAS调用可能产生的异步CUDA错误
    
    // 计时结束
    checkCuda(cudaEventRecord(stop_gpu_cublas));
    checkCuda(cudaEventSynchronize(stop_gpu_cublas)); // 等待核函数完成

    float ms_cublas = 0;
    checkCuda(cudaEventElapsedTime(&ms_cublas, start_gpu_cublas, stop_gpu_cublas));
    std::cout << "GPU (cuBLAS) execution finished. Time: " << ms_cublas << " ms" << std::endl;

    if (ms_shared_mem > 1e-9) { // 避免除以零或非常小的值
        float speedup_cublas_over_shared = ms_shared_mem / ms_cublas;
        std::cout << "Speedup of cuBLAS over Shared Memory: " << speedup_cublas_over_shared << "x" << std::endl;
    }
    if (ms_global_mem > 1e-9) {
        float speedup_cublas_over_global = ms_global_mem / ms_cublas;
        std::cout << "Speedup of cuBLAS over Global Memory: " << speedup_cublas_over_global << "x" << std::endl;
    }

    checkCuda(cudaEventDestroy(start_gpu_cublas));
    checkCuda(cudaEventDestroy(stop_gpu_cublas));
    checkCublas(cublasDestroy(cublas_handle));

    // 将结果从设备复制回主机
    checkCuda(cudaMemcpy(h_output_gpu_cublas.data(), d_output_cublas, dataSize, cudaMemcpyDeviceToHost));

    // --- 验证结果 ---
    std::cout << "Verifying results..." << std::endl;
    bool global_mem_success = true;
    bool shared_mem_success = true;
    bool cublas_success = true; // 为cuBLAS版本添加
    
    // 验证前100个元素是否正确
    int check_count = std::min(100, N * N);
    for (int i = 0; i < check_count; ++i) {
        if (global_mem_success && std::abs(h_output_cpu[i] - h_output_gpu_global[i]) > 1e-3) {
            std::cerr << "Mismatch (Global Memory) at index " << i << ": CPU=" << h_output_cpu[i] << ", GPU_Global=" << h_output_gpu_global[i] << std::endl;
            global_mem_success = false;
            // 不再在此处 break，以便检查所有版本的前 check_count 个元素
        }
        if (shared_mem_success && std::abs(h_output_cpu[i] - h_output_gpu_shared[i]) > 1e-3) {
            std::cerr << "Mismatch (Shared Memory) at index " << i << ": CPU=" << h_output_cpu[i] << ", GPU_Shared=" << h_output_gpu_shared[i] << std::endl;
            shared_mem_success = false;
        }
        if (cublas_success && std::abs(h_output_cpu[i] - h_output_gpu_cublas[i]) > 1e-3) { // 为cuBLAS版本添加
            std::cerr << "Mismatch (cuBLAS) at index " << i << ": CPU=" << h_output_cpu[i] << ", GPU_cuBLAS=" << h_output_gpu_cublas[i] << std::endl;
            cublas_success = false;
        }
    }
    
    // 对于更大的矩阵，需要使用更合理的验证方法
    double sum_cpu = 0.0, sum_gpu_global = 0.0, sum_gpu_shared = 0.0, sum_gpu_cublas = 0.0; // 为cuBLAS版本添加
    // 只采样部分元素求和，避免求和过程中的精度损失
    int stride = N * N / 1000; // 每1000个元素采样一个进行验证
    if (stride < 1) stride = 1;

    for (int i = 0; i < N * N; i += stride) {
        sum_cpu += h_output_cpu[i];
        sum_gpu_global += h_output_gpu_global[i];
        sum_gpu_shared += h_output_gpu_shared[i];
        sum_gpu_cublas += h_output_gpu_cublas[i]; // 为cuBLAS版本添加
    }

    // 使用相对误差而不是绝对误差
    if (sum_cpu != 0) { // 避免除以零
        double rel_error_global = std::abs((sum_cpu - sum_gpu_global) / sum_cpu);
        double rel_error_shared = std::abs((sum_cpu - sum_gpu_shared) / sum_cpu);
        double rel_error_cublas = std::abs((sum_cpu - sum_gpu_cublas) / sum_cpu); // 为cuBLAS版本添加

        if (rel_error_global > 1e-3) {
            global_mem_success = false;
            std::cerr << "Sum mismatch (Global Memory): CPU=" << sum_cpu << ", GPU_Global=" << sum_gpu_global 
                      << ", Relative Error=" << rel_error_global << std::endl;
        }
        
        if (rel_error_shared > 1e-3) {
            shared_mem_success = false;
            std::cerr << "Sum mismatch (Shared Memory): CPU=" << sum_cpu << ", GPU_Shared=" << sum_gpu_shared 
                      << ", Relative Error=" << rel_error_shared << std::endl;
        }
        if (rel_error_cublas > 1e-3) { // 为cuBLAS版本添加
            cublas_success = false;
            std::cerr << "Sum mismatch (cuBLAS): CPU=" << sum_cpu << ", GPU_cuBLAS=" << sum_gpu_cublas 
                      << ", Relative Error=" << rel_error_cublas << std::endl;
        }
    } else { // 如果 sum_cpu 为零，则检查绝对差值
        if (std::abs(sum_cpu - sum_gpu_global) > 1e-3 * (N*N/stride)) global_mem_success = false;
        if (std::abs(sum_cpu - sum_gpu_shared) > 1e-3 * (N*N/stride)) shared_mem_success = false;
        if (std::abs(sum_cpu - sum_gpu_cublas) > 1e-3 * (N*N/stride)) cublas_success = false; // 为cuBLAS版本添加
         if (!global_mem_success) std::cerr << "Sum mismatch (Global Memory): CPU=" << sum_cpu << ", GPU_Global=" << sum_gpu_global << std::endl;
         if (!shared_mem_success) std::cerr << "Sum mismatch (Shared Memory): CPU=" << sum_cpu << ", GPU_Shared=" << sum_gpu_shared << std::endl;
         if (!cublas_success) std::cerr << "Sum mismatch (cuBLAS): CPU=" << sum_cpu << ", GPU_cuBLAS=" << sum_gpu_cublas << std::endl;
    }
    
    if (global_mem_success) {
        std::cout << "Validation passed: Global memory version results are correct." << std::endl;
    } else {
        std::cout << "Validation failed: Global memory version results are incorrect!" << std::endl;
    }
    
    if (shared_mem_success) {
        std::cout << "Validation passed: Shared memory version results are correct." << std::endl;
    } else {
        std::cout << "Validation failed: Shared memory version results are incorrect!" << std::endl;
    }

    if (cublas_success) { // 为cuBLAS版本添加
        std::cout << "Validation passed: cuBLAS version results are correct." << std::endl;
    } else {
        std::cout << "Validation failed: cuBLAS version results are incorrect!" << std::endl;
    }

    // --- 清理 ---
    std::cout << "Cleaning up GPU memory..." << std::endl;
    cudaFree(d_input);
    cudaFree(d_output_global);
    cudaFree(d_output_shared);
    cudaFree(d_output_cublas); // 为cuBLAS版本添加
    std::cout << "GPU memory released." << std::endl;

    // --- 共享内存在矩阵转置中的优势 ---
    // 1. 矩阵转置问题的特点：
    //    - 全局内存版本中，读取是合并的，但写入是不合并的
    //    - 这导致全局内存带宽利用率低下，性能受限
    // 2. 共享内存如何提高性能：
    //    - 先以合并方式读取输入矩阵到共享内存
    //    - 在共享内存中完成转置操作（这是快速的片上操作）
    //    - 然后以合并方式写回全局内存
    // 3. 共享内存的关键优势：
    //    - 避免了全局内存的不合并访问
    //    - 降低了内存延迟，提高了带宽利用率
    //    - 通过转换访问模式优化了内存访问合并度
    // 4. 共享内存访问模式优化：
    //    - 共享内存中加入了一行填充避免了共享内存的银行冲突
    //    - 利用矩阵分块和线程块结构实现高效读写模式
    // 5. 性能提升因素：
    //    - 消除了不合并的全局内存访问
    //    - 充分利用了CUDA硬件的内存层次结构
    //    - 提高了内存访问的并行度

    std::cout << "\nMatrix transpose comparison (Global, Shared, and cuBLAS versions) complete." << std::endl;
    return 0;
} 