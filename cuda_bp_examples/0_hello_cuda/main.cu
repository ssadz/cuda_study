#include <iostream>     // 用于标准输入输出，比如 std::cout
#include <cuda_runtime.h> // CUDA 运行时 API 的头文件，包含了大部分 CUDA 函数声明

// CUDA 核函数 (Kernel) - 这部分代码会在 GPU 上并行执行
// __global__ 关键字表示这是一个可以从 CPU 调用并在 GPU 上执行的函数
__global__ void helloFromGPU() {
    // threadIdx.x 是当前线程在块内的 x 维度索引
    // blockIdx.x 是当前块在网格内的 x 维度索引
    // 这个判断确保 "Hello World" 只被每个块的第一个线程打印一次，避免重复输出
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("你好，来自 GPU 的世界!\n"); // GPU 上的 printf，输出会显示在控制台
    }
}

// 主函数 - 程序从这里开始执行 (在 CPU 上运行)
int main() {
    std::cout << "你好，来自 CPU 的世界!" << std::endl;

    // 检查系统中是否存在 CUDA 设备
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount); // 获取 CUDA 设备数量

    // cudaSuccess 是一个枚举值，表示 CUDA API 调用成功
    if (err != cudaSuccess || deviceCount == 0) {
        std::cout << "未找到 CUDA 设备或 CUDA 初始化错误: " << cudaGetErrorString(err) << std::endl;
        return 1; // 如果没有设备或出错，则退出程序
    }
    std::cout << "发现 " << deviceCount << " 个 CUDA 设备." << std::endl;

    // 选择第一个 CUDA 设备 (设备编号从 0 开始)
    // 对于多 GPU 系统，你可以选择特定的 GPU
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cout << "设置 CUDA 设备失败: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "正在使用设备 0." << std::endl;

    // 启动 CUDA 核函数
    // <<<...>>> 是 CUDA 的核函数启动语法
    // 第一个参数: 网格维度 (Grid Dimension)，这里是 1 个块
    // 第二个参数: 块维度 (Block Dimension)，这里是 1 个线程
    // 总共启动 1 * 1 = 1 个线程
    helloFromGPU<<<1, 1>>>();

    // 等待 GPU 完成所有之前提交的任务
    // 这很重要，因为核函数启动是异步的。不等待的话，CPU 上的主函数可能在 GPU printf 执行完之前就退出了。
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "核函数启动或同步失败: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "GPU 核函数执行完毕." << std::endl;
    return 0; // 程序正常结束
}