#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

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

// 核函数1: 通过全局内存直接读取
__global__ void globalMemoryAccessKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 模拟一些重复或具有一定局部性的访问
        float sum = 0.0f;
        for (int i = -2; i <= 2; ++i) {
            int access_idx = idx + i;
            if (access_idx >= 0 && access_idx < n) {
                sum += input[access_idx];
            }
        }
        output[idx] = sum / 5.0f; // 简单的移动平均
    }
}

// 核函数2: 通过纹理对象读取
__global__ void textureObjectAccessKernel(cudaTextureObject_t texObj, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 使用 tex1Dfetch 从纹理对象中读取数据
        // 纹理内存通常对具有空间局部性的访问有益
        float sum = 0.0f;
        for (int i = -2; i <= 2; ++i) {
            int access_idx = idx + i;
            if (access_idx >= 0 && access_idx < n) {
                sum += tex1Dfetch<float>(texObj, access_idx);
            }
        }
        output[idx] = sum / 5.0f; // 简单的移动平均
    }
}

// 核函数3: 演示纹理的线性插值功能 (使用纹理对象)
// 要使用硬件线性插值，纹理描述符中需要设置 filterMode 为 cudaFilterModeLinear，
// 并且使用 tex1D() 等函数配合浮点坐标。
// tex1Dfetch() 通常用于整数坐标，不直接进行插值。
// 为了简化，此核函数仍使用 tex1Dfetch，并进行手动插值。
// 一个更佳的插值演示会使用 tex1D() 和浮点坐标。
__global__ void textureObjectInterpolationKernel(cudaTextureObject_t texObjLerp, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n -1 && idx > 0) { // 避免边界
        // 假设我们要读取索引为 idx + 0.5f 的位置
        // 使用 tex1Dfetch 获取左右两边的值
        float val_left = tex1Dfetch<float>(texObjLerp, idx);
        float val_right = tex1Dfetch<float>(texObjLerp, idx + 1);
        output[idx] = 0.5f * val_left + 0.5f * val_right; // 手动线性插值
    } else if (idx < n) {
        output[idx] = tex1Dfetch<float>(texObjLerp, idx); // 边界处直接读取
    }
}


int main() {
    // 字符串字面量保持英文
    std::cout << "CUDA Texture Object Example (Best Practices Chapter 12.2.5)" << std::endl;

    const int N = 1024 * 1024 * 4; // 4M 元素
    const size_t dataSize = N * sizeof(float);

    // --- 主机数据准备 ---
    std::vector<float> h_input(N);
    std::vector<float> h_output_global(N, 0.0f);
    std::vector<float> h_output_texture(N, 0.0f);
    std::vector<float> h_output_lerp(N, 0.0f);
    std::vector<float> h_expected_avg(N);
    std::vector<float> h_expected_lerp(N);

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 100); // 简单的周期数据
    }

    // 计算期望结果 (移动平均)
    for (int idx = 0; idx < N; ++idx) {
        float sum = 0.0f;
        int count = 0;
        for (int i = -2; i <= 2; ++i) {
            int access_idx = idx + i;
            if (access_idx >= 0 && access_idx < N) {
                sum += h_input[access_idx];
                count++;
            }
        }
        h_expected_avg[idx] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;

        // 计算期望结果 (手动线性插值)
        if (idx < N - 1 && idx > 0) {
            h_expected_lerp[idx] = 0.5f * h_input[idx] + 0.5f * h_input[idx+1];
        } else {
            h_expected_lerp[idx] = h_input[idx];
        }
    }

    // --- 设备数据 ---
    float* d_input = nullptr;
    float* d_output_global = nullptr;
    float* d_output_texture = nullptr;
    float* d_output_lerp_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, dataSize));
    CUDA_CHECK(cudaMalloc(&d_output_global, dataSize));
    CUDA_CHECK(cudaMalloc(&d_output_texture, dataSize));
    CUDA_CHECK(cudaMalloc(&d_output_lerp_dev, dataSize));

    // 将输入数据拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), dataSize, cudaMemcpyHostToDevice));

    // --- 纹理对象设置 ---
    cudaTextureObject_t texObjInput = 0;
    cudaTextureObject_t texObjLerp = 0;

    // 1. 为 texObjInput 创建资源描述符
    cudaResourceDesc resDescInput;
    memset(&resDescInput, 0, sizeof(resDescInput));
    resDescInput.resType = cudaResourceTypeLinear; // 线性内存
    resDescInput.res.linear.devPtr = d_input;
    resDescInput.res.linear.desc.f = cudaChannelFormatKindFloat; // 数据类型
    resDescInput.res.linear.desc.x = 32; // bits per channel
    resDescInput.res.linear.sizeInBytes = dataSize;

    // 2. 为 texObjInput 创建纹理描述符
    cudaTextureDesc texDescInput;
    memset(&texDescInput, 0, sizeof(texDescInput));
    texDescInput.addressMode[0] = cudaAddressModeClamp; // 寻址模式 (X方向)
    // texDescInput.addressMode[1] = cudaAddressModeClamp; // 寻址模式 (Y方向, 1D纹理不需要)
    texDescInput.filterMode = cudaFilterModePoint;     // 滤波模式 (点采样)
    texDescInput.readMode = cudaReadModeElementType;   // 读取模式 (直接读取元素类型)
    texDescInput.normalizedCoords = 0;                 // 不使用归一化坐标

    // 3. 创建纹理对象 texObjInput
    CUDA_CHECK(cudaCreateTextureObject(&texObjInput, &resDescInput, &texDescInput, nullptr));

    // 4. 为 texObjLerp 创建资源和纹理描述符 (与 texObjInput 类似，因为它们使用相同的源数据)
    cudaResourceDesc resDescLerp = resDescInput; // 可以复用
    cudaTextureDesc texDescLerp = texDescInput;
    // 如果要演示硬件插值，需要修改 texDescLerp:
    // texDescLerp.filterMode = cudaFilterModeLinear;
    // texDescLerp.normalizedCoords = 1; // 或者使用非归一化浮点坐标
    // texDescLerp.addressMode[0] = cudaAddressModeBorder; // 使用边界颜色可能更好
    // float borderColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
    // texDescLerp.borderColor[0] = borderColor[0]; ...

    CUDA_CHECK(cudaCreateTextureObject(&texObjLerp, &resDescLerp, &texDescLerp, nullptr));


    // CUDA 执行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    float ms;

    // --- 1. 测试全局内存访问核函数 ---
    // 字符串字面量保持英文
    std::cout << "\n--- Testing Global Memory Access Kernel ---" << std::endl;
    CUDA_CHECK(cudaMemset(d_output_global, 0, dataSize));

    CUDA_CHECK(cudaEventRecord(start_event));
    globalMemoryAccessKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output_global, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "Global Memory Access Kernel Time: " << ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_output_global.data(), d_output_global, dataSize, cudaMemcpyDeviceToHost));
    bool global_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_output_global[i] - h_expected_avg[i]) > 1e-4) { global_ok = false; break;}
    // 字符串字面量保持英文
    std::cout << "Global Memory Access Verification: " << (global_ok ? "PASSED" : "FAILED") << std::endl;

    // --- 2. 测试纹理对象访问核函数 ---
    // 字符串字面量保持英文
    std::cout << "\n--- Testing Texture Object Access Kernel ---" << std::endl;
    CUDA_CHECK(cudaMemset(d_output_texture, 0, dataSize));

    CUDA_CHECK(cudaEventRecord(start_event));
    textureObjectAccessKernel<<<blocksPerGrid, threadsPerBlock>>>(texObjInput, d_output_texture, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "Texture Object Access Kernel Time: " << ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_output_texture.data(), d_output_texture, dataSize, cudaMemcpyDeviceToHost));
    bool texture_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_output_texture[i] - h_expected_avg[i]) > 1e-4) { texture_ok = false; break;}
    // 字符串字面量保持英文
    std::cout << "Texture Object Access Verification: " << (texture_ok ? "PASSED" : "FAILED") << std::endl;

    // --- 3. 测试纹理插值核函数 (使用纹理对象, 手动模拟) ---
    // 字符串字面量保持英文
    std::cout << "\n--- Testing Texture Object Interpolation Kernel (Manual LERP) ---" << std::endl;
    CUDA_CHECK(cudaMemset(d_output_lerp_dev, 0, dataSize));

    CUDA_CHECK(cudaEventRecord(start_event));
    textureObjectInterpolationKernel<<<blocksPerGrid, threadsPerBlock>>>(texObjLerp, d_output_lerp_dev, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    // 字符串字面量保持英文
    std::cout << "Texture Object Interpolation Kernel Time: " << ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_output_lerp.data(), d_output_lerp_dev, dataSize, cudaMemcpyDeviceToHost));
    bool lerp_ok = true;
    for(int i=0; i<N; ++i) if(std::abs(h_output_lerp[i] - h_expected_lerp[i]) > 1e-4) { lerp_ok = false; break; }
    // 字符串字面量保持英文
    std::cout << "Texture Object Interpolation Verification: " << (lerp_ok ? "PASSED" : "FAILED") << std::endl;


    // --- 教学要点 ---
    // 字符串字面量保持英文 (std::cout 部分)
    std::cout << "\nTeaching Points (Best Practices Guide - Chapter 12.2.5 Texture Memory & Texture Objects):" << std::endl;
    // 中文注释开始 (对下面英文教学点的解释)
    // 1. 纹理对象 (`cudaTextureObject_t`) 是现代 CUDA 中使用纹理内存的首选方式。
    // 2. 创建纹理对象需要 `cudaResourceDesc` (描述数据源，如线性内存或 CUDA 数组) 和 `cudaTextureDesc` (描述采样行为，如寻址模式、滤波模式)。
    // 3. 使用 `cudaCreateTextureObject()` 创建纹理对象，并在不再需要时用 `cudaDestroyTextureObject()` 销毁。
    // 4. 核函数通过接收 `cudaTextureObject_t` 类型的参数来访问纹理。
    // 5. `tex1Dfetch<ReturnType>(texObject, int coord)` 用于从一维纹理对象中获取数据。注意模板参数 `ReturnType`。
    // 6. 纹理内存的优势（如缓存、硬件插值、边界处理）通过正确配置 `cudaTextureDesc` 来实现。
    // 7. 对于硬件线性插值，通常需要将 `filterMode` 设置为 `cudaFilterModeLinear`，并使用 `tex1D()` 等函数配合浮点坐标。本例中的插值是手动的。
    // 8. 寻址模式（如 `cudaAddressModeClamp`, `cudaAddressModeWrap`, `cudaAddressModeBorder`）控制当纹理坐标超出 `[0, 纹理维度-1]` 或 `[0.0, 1.0]` (对于归一化坐标) 范围时的行为。
    // 中文注释结束
    std::cout << "1. Texture objects (`cudaTextureObject_t`) are the modern way to use texture memory in CUDA." << std::endl;
    std::cout << "2. Creating a texture object requires a `cudaResourceDesc` (describes data source) and `cudaTextureDesc` (describes sampling behavior)." << std::endl;
    std::cout << "3. Use `cudaCreateTextureObject()` to create and `cudaDestroyTextureObject()` to destroy them." << std::endl;
    std::cout << "4. Kernels access textures by receiving the `cudaTextureObject_t` as an argument." << std::endl;
    std::cout << "5. `tex1Dfetch<ReturnType>(texObject, int coord)` fetches data from a 1D texture object. Note the `ReturnType` template argument." << std::endl;
    std::cout << "6. Texture memory benefits (caching, hardware interpolation, boundary modes) are realized by proper `cudaTextureDesc` configuration." << std::endl;
    std::cout << "7. For hardware linear interpolation, `filterMode` should be `cudaFilterModeLinear`, and functions like `tex1D()` with float coordinates are used. Interpolation in this example is manual." << std::endl;
    std::cout << "8. Addressing modes (e.g., `cudaAddressModeClamp`, `cudaAddressModeWrap`, `cudaAddressModeBorder`) control behavior for out-of-bounds coordinates." << std::endl;

    // --- 清理 ---
    CUDA_CHECK(cudaDestroyTextureObject(texObjInput));
    CUDA_CHECK(cudaDestroyTextureObject(texObjLerp));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_global));
    CUDA_CHECK(cudaFree(d_output_texture));
    CUDA_CHECK(cudaFree(d_output_lerp_dev));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    // 字符串字面量保持英文
    std::cout << "\nTexture object example finished." << std::endl;
    return 0;
}