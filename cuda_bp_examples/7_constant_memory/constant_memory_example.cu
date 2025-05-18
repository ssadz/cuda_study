#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cmath>

// 检查CUDA错误的辅助宏
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(err_) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 定义数组大小和常量内存数组
constexpr int DATA_SIZE = 1 << 20;  // 1M elements
constexpr int FILTER_SIZE = 128;    // 滤波器大小

// 声明常量内存数组（只能在全局作用域声明）
__constant__ float constFilter[FILTER_SIZE];

// 使用全局内存的卷积核函数
__global__ void convolutionGlobal(const float* input, float* output, 
                                 const float* filter, int dataSize, int filterSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        float result = 0.0f;
        
        // 对于每个元素，应用滤波器
        for (int i = 0; i < filterSize; i++) {
            int offset = idx - filterSize / 2 + i;
            // 边界检查
            if (offset >= 0 && offset < dataSize) {
                result += input[offset] * filter[i];
            }
        }
        
        output[idx] = result;
    }
}

// 使用常量内存的卷积核函数
__global__ void convolutionConstant(const float* input, float* output, 
                                  int dataSize, int filterSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        float result = 0.0f;
        
        // 对于每个元素，应用从常量内存读取的滤波器
        for (int i = 0; i < filterSize; i++) {
            int offset = idx - filterSize / 2 + i;
            // 边界检查
            if (offset >= 0 && offset < dataSize) {
                result += input[offset] * constFilter[i];
            }
        }
        
        output[idx] = result;
    }
}

int main() {
    // 分配主机内存
    float* h_input = new float[DATA_SIZE];
    float* h_filter = new float[FILTER_SIZE];
    float* h_output_global = new float[DATA_SIZE];
    float* h_output_constant = new float[DATA_SIZE];
    
    // 初始化输入数据和滤波器
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = static_cast<float>(sinf(i) * 10.0f);
    }
    
    for (int i = 0; i < FILTER_SIZE; i++) {
        // 创建一个高斯滤波器
        float x = i - FILTER_SIZE / 2;
        h_filter[i] = expf(-(x*x) / (2 * 15 * 15));
    }
    
    // 分配设备内存
    float *d_input, *d_filter, *d_output_global, *d_output_constant;
    CUDA_CHECK(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, FILTER_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_global, DATA_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_constant, DATA_SIZE * sizeof(float)));
    
    // 将数据从主机拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // 将滤波器拷贝到常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(constFilter, h_filter, FILTER_SIZE * sizeof(float)));
    
    // 设置执行配置
    int blockSize = 256;
    int gridSize = (DATA_SIZE + blockSize - 1) / blockSize;
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float msGlobal = 0.0f, msConstant = 0.0f;
    
    // 运行使用全局内存的卷积
    CUDA_CHECK(cudaEventRecord(start));
    convolutionGlobal<<<gridSize, blockSize>>>(d_input, d_output_global, d_filter, DATA_SIZE, FILTER_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msGlobal, start, stop));
    
    // 运行使用常量内存的卷积
    CUDA_CHECK(cudaEventRecord(start));
    convolutionConstant<<<gridSize, blockSize>>>(d_input, d_output_constant, DATA_SIZE, FILTER_SIZE);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msConstant, start, stop));
    
    // 将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_output_global, d_output_global, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_constant, d_output_constant, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 验证结果是否一致
    float maxError = 0.0f;
    for (int i = 0; i < DATA_SIZE; i++) {
        maxError = fmax(maxError, fabs(h_output_global[i] - h_output_constant[i]));
    }
    
    // 输出性能比较结果
    std::cout << "Convolution using global memory: " << msGlobal << " ms" << std::endl;
    std::cout << "Convolution using constant memory: " << msConstant << " ms" << std::endl;
    std::cout << "Performance improvement: " << (msGlobal / msConstant) << "x" << std::endl;
    std::cout << "Max error between results: " << maxError << std::endl;
    
    // 释放内存
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output_global;
    delete[] h_output_constant;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output_global));
    CUDA_CHECK(cudaFree(d_output_constant));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
} 