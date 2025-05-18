#include <iostream>
#include <vector>
#include <numeric>      // For std::iota
#include <algorithm>    // For std::equal, std::fill
#include <chrono>       // For timing (optional)
#include <cmath>        // For std::abs

// 辅助函数，用于检查 CUDA API 调用是否成功
// 这是一个好习惯，可以帮助快速定位运行时错误。
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA 运行时错误: " << cudaGetErrorString(result) << std::endl;
        // 在实际应用中，你可能希望在这里抛出异常或采取更复杂的错误处理策略
    }
    return result;
}

// 一个简单的 CUDA 核函数，将数组中的每个元素加倍
// a: 指向设备内存中数组的指针
// n: 数组中的元素数量
__global__ void doubleElements(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= 2.0f;
    }
}

// 一个简单的 CUDA 核函数，将数组中的每个元素增加一个常量值
__global__ void incrementElements(float* a, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += value;
    }
}


int main() {
    std::cout << "CUDA 内存管理教学示例 (第三章最佳实践)" << std::endl;

    const int N = 1024 * 1024; // 数组大小 (1M 元素)
    const size_t dataSize = N * sizeof(float); // 数据总字节数

    // --- 主机 (CPU) 数据准备 ---
    std::vector<float> h_data_original(N);
    std::vector<float> h_data_result(N);
    std::vector<float> h_data_expected(N); // 用于验证

    // 初始化原始数据
    std::iota(h_data_original.begin(), h_data_original.end(), 1.0f); // {1.0f, 2.0f, ..., N.0f}
    // 计算期望结果 (每个元素加倍)
    for (int i = 0; i < N; ++i) {
        h_data_expected[i] = h_data_original[i] * 2.0f;
    }

    // CUDA 执行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // =========================================================================
    // 1. 基本设备内存管理 (cudaMalloc, cudaFree, cudaMemcpy)
    // =========================================================================
    std::cout << "--- 1. 基本设备内存管理 ---" << std::endl;
    {
        // 清空结果容器，以便本节使用
        std::fill(h_data_result.begin(), h_data_result.end(), 0.0f);

        float* d_data = nullptr; // 设备内存指针

        // 1.1 在 GPU 设备上分配内存
        // cudaMalloc: 分配全局内存。这是最常见的设备内存分配方式。
        std::cout << "  分配设备内存 (d_data)... ";
        if (checkCuda(cudaMalloc((void**)&d_data, dataSize)) != cudaSuccess) return 1;
        std::cout << "完成." << std::endl;

        // 1.2 将数据从主机内存复制到设备内存 (Host-to-Device)
        // cudaMemcpy: 同步内存复制。函数调用会阻塞，直到复制完成。
        // cudaMemcpyHostToDevice: 明确指定复制方向。
        std::cout << "  将数据从主机复制到设备 (h_data_original -> d_data)... ";
        if (checkCuda(cudaMemcpy(d_data, h_data_original.data(), dataSize, cudaMemcpyHostToDevice)) != cudaSuccess) {
            checkCuda(cudaFree(d_data)); // 清理
            return 1;
        }
        std::cout << "完成." << std::endl;

        // 1.3 在 GPU 上执行核函数
        std::cout << "  启动 `doubleElements` 核函数... ";
        doubleElements<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
        // 检查核函数启动错误 (异步错误需要 cudaDeviceSynchronize 后检查，或使用 cudaGetLastError)
        if (checkCuda(cudaGetLastError()) != cudaSuccess) { /* 清理 */ checkCuda(cudaFree(d_data)); return 1; }
        // 同步设备以确保核函数执行完成
        if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess) { /* 清理 */ checkCuda(cudaFree(d_data)); return 1; }
        std::cout << "完成." << std::endl;

        // 1.4 将结果从设备内存复制回主机内存 (Device-to-Host)
        std::cout << "  将结果从设备复制回主机 (d_data -> h_data_result)... ";
        if (checkCuda(cudaMemcpy(h_data_result.data(), d_data, dataSize, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            checkCuda(cudaFree(d_data)); // 清理
            return 1;
        }
        std::cout << "完成." << std::endl;

        // 1.5 验证结果
        if (std::equal(h_data_result.begin(), h_data_result.end(), h_data_expected.begin())) {
            std::cout << "  结果验证成功!" << std::endl;
        } else {
            std::cout << "  结果验证失败!" << std::endl;
        }

        // 1.6 释放设备内存
        // cudaFree: 释放之前通过 cudaMalloc 分配的设备内存。
        // 忘记释放内存会导致内存泄漏。
        std::cout << "  释放设备内存 (d_data)... ";
        checkCuda(cudaFree(d_data));
        std::cout << "完成." << std::endl;
    }
    std::cout << std::endl;


    // =========================================================================
    // 2. 固定主机内存 (Pinned/Page-Locked Memory)
    //    - 使用 cudaMallocHost, cudaFreeHost
    //    - 固定内存允许 GPU 通过直接内存访问 (DMA) 进行读写，
    //      从而实现更高效的异步内存传输。
    //    - 对于 cudaMemcpyAsync，源内存和目标内存中至少有一个必须是固定的。
    //      如果源是主机内存而目标是设备内存，则主机内存必须是固定的。
    // =========================================================================
    std::cout << "--- 2. 固定主机内存 (Pinned Memory) ---" << std::endl;
    {
        // 清空结果容器
        std::fill(h_data_result.begin(), h_data_result.end(), 0.0f);
        
        // 计算期望结果 (每个元素增加 5.0f)
        std::vector<float> h_data_expected_pinned(N);
        for (int i = 0; i < N; ++i) {
            h_data_expected_pinned[i] = h_data_original[i] + 5.0f;
        }

        float* h_pinned_data_in = nullptr;  // 指向主机固定输入内存的指针
        float* h_pinned_data_out = nullptr; // 指向主机固定输出内存的指针
        float* d_data_pinned_test = nullptr; // 设备内存指针

        // 2.1 分配固定主机内存
        // cudaMallocHost: 分配页面锁定的主机内存。
        // 注意：过度使用固定内存会减少可用于操作系统的可分页内存，可能导致系统性能下降。
        std::cout << "  分配固定主机内存 (h_pinned_data_in, h_pinned_data_out)... ";
        if (checkCuda(cudaMallocHost((void**)&h_pinned_data_in, dataSize)) != cudaSuccess) return 1;
        if (checkCuda(cudaMallocHost((void**)&h_pinned_data_out, dataSize)) != cudaSuccess) {
            checkCuda(cudaFreeHost(h_pinned_data_in)); return 1;
        }
        std::cout << "完成." << std::endl;

        // 初始化固定的输入数据
        for (int i = 0; i < N; ++i) {
            h_pinned_data_in[i] = h_data_original[i];
        }
        std::fill_n(h_pinned_data_out, N, 0.0f); // 清零输出缓冲区

        // 2.2 分配设备内存
        std::cout << "  分配设备内存 (d_data_pinned_test)... ";
        if (checkCuda(cudaMalloc((void**)&d_data_pinned_test, dataSize)) != cudaSuccess) {
            checkCuda(cudaFreeHost(h_pinned_data_in)); checkCuda(cudaFreeHost(h_pinned_data_out)); return 1;
        }
        std::cout << "完成." << std::endl;

        // 2.3 使用固定内存进行同步数据传输 (仍然可以使用 cudaMemcpy)
        std::cout << "  使用固定内存进行同步复制 (h_pinned_data_in -> d_data_pinned_test)... ";
        if (checkCuda(cudaMemcpy(d_data_pinned_test, h_pinned_data_in, dataSize, cudaMemcpyHostToDevice)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;
        
        // 2.4 执行核函数 (这里用 incrementElements 区分)
        std::cout << "  启动 `incrementElements` 核函数 (value=5.0f)... ";
        incrementElements<<<blocksPerGrid, threadsPerBlock>>>(d_data_pinned_test, N, 5.0f);
        if (checkCuda(cudaGetLastError()) != cudaSuccess) { /* 清理 */ }
        if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 2.5 使用固定内存从设备同步复制回主机
        std::cout << "  使用固定内存从设备同步复制回主机 (d_data_pinned_test -> h_pinned_data_out)... ";
        if (checkCuda(cudaMemcpy(h_pinned_data_out, d_data_pinned_test, dataSize, cudaMemcpyDeviceToHost)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 2.6 验证结果 (h_pinned_data_out)
        bool pinned_sync_success = true;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_pinned_data_out[i] - h_data_expected_pinned[i]) > 1e-5) {
                pinned_sync_success = false;
                break;
            }
        }
        if (pinned_sync_success) {
            std::cout << "  固定内存同步传输结果验证成功!" << std::endl;
        } else {
            std::cout << "  固定内存同步传输结果验证失败!" << std::endl;
        }

        // 2.7 释放固定主机内存和设备内存
        // cudaFreeHost: 释放通过 cudaMallocHost 分配的内存。
        std::cout << "  释放固定主机内存和设备内存... ";
        checkCuda(cudaFreeHost(h_pinned_data_in));
        checkCuda(cudaFreeHost(h_pinned_data_out));
        checkCuda(cudaFree(d_data_pinned_test));
        std::cout << "完成." << std::endl;
    }
    std::cout << std::endl;

    // =========================================================================
    // 3. 异步内存拷贝与流 (Asynchronous Memory Copies and Streams)
    //    - cudaMemcpyAsync 需要固定主机内存。
    //    - 流 (cudaStream_t) 允许将操作（如内存拷贝、核函数启动）排入队列，
    //      从而实现主机、设备计算以及不同设备操作之间的重叠执行。
    // =========================================================================
    std::cout << "--- 3. 异步内存拷贝与流 ---" << std::endl;
    {
        // 清空结果容器
        std::fill(h_data_result.begin(), h_data_result.end(), 0.0f);
        // 期望结果与第1部分相同 (doubleElements)
        
        float* h_pinned_async_in = nullptr;
        float* h_pinned_async_out = nullptr;
        float* d_data_async_test = nullptr;
        cudaStream_t stream; // CUDA 流

        // 3.1 创建 CUDA 流
        std::cout << "  创建 CUDA 流... ";
        if (checkCuda(cudaStreamCreate(&stream)) != cudaSuccess) return 1;
        std::cout << "完成." << std::endl;

        // 3.2 分配固定主机内存
        std::cout << "  分配固定主机内存 (h_pinned_async_in, h_pinned_async_out)... ";
        if (checkCuda(cudaMallocHost((void**)&h_pinned_async_in, dataSize)) != cudaSuccess) { /* 清理 */ }
        if (checkCuda(cudaMallocHost((void**)&h_pinned_async_out, dataSize)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 初始化固定的输入数据
        for (int i = 0; i < N; ++i) {
             h_pinned_async_in[i] = h_data_original[i];
        }
        std::fill_n(h_pinned_async_out, N, 0.0f);

        // 3.3 分配设备内存
        std::cout << "  分配设备内存 (d_data_async_test)... ";
        if (checkCuda(cudaMalloc((void**)&d_data_async_test, dataSize)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 3.4 异步将数据从主机复制到设备
        // cudaMemcpyAsync: 将内存复制操作排入指定的流。调用立即返回。
        // 主机内存 h_pinned_async_in 必须是固定的。
        std::cout << "  异步复制数据到设备 (h_pinned_async_in -> d_data_async_test) 到流中... ";
        if (checkCuda(cudaMemcpyAsync(d_data_async_test, h_pinned_async_in, dataSize, cudaMemcpyHostToDevice, stream)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 3.5 将核函数启动排入同一个流
        // 核函数启动默认是异步的，可以指定流以与其他操作同步。
        std::cout << "  启动 `doubleElements` 核函数到流中... ";
        doubleElements<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data_async_test, N);
        if (checkCuda(cudaGetLastError()) != cudaSuccess) { /* 清理 */ } // 检查启动错误
        std::cout << "完成." << std::endl;

        // 3.6 异步将结果从设备复制回主机
        // 主机内存 h_pinned_async_out 必须是固定的。
        std::cout << "  异步复制结果回主机 (d_data_async_test -> h_pinned_async_out) 到流中... ";
        if (checkCuda(cudaMemcpyAsync(h_pinned_async_out, d_data_async_test, dataSize, cudaMemcpyDeviceToHost, stream)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 3.7 同步流
        // cudaStreamSynchronize: 阻塞主机线程，直到指定流中的所有先前操作完成。
        std::cout << "  同步流 (等待所有异步操作完成)... ";
        if (checkCuda(cudaStreamSynchronize(stream)) != cudaSuccess) { /* 清理 */ }
        std::cout << "完成." << std::endl;

        // 3.8 验证结果
        if (std::equal(h_pinned_async_out, h_pinned_async_out + N, h_data_expected.begin())) {
            std::cout << "  异步传输结果验证成功!" << std::endl;
        } else {
            std::cout << "  异步传输结果验证失败!" << std::endl;
        }
        
        // 3.9 销毁流并释放内存
        std::cout << "  销毁流并释放内存... ";
        checkCuda(cudaStreamDestroy(stream));
        checkCuda(cudaFreeHost(h_pinned_async_in));
        checkCuda(cudaFreeHost(h_pinned_async_out));
        checkCuda(cudaFree(d_data_async_test));
        std::cout << "完成." << std::endl;
    }
    std::cout << std::endl;


    // =========================================================================
    // 4. 统一内存 (Unified Memory)
    //    - 使用 cudaMallocManaged 分配。
    //    - 创建一个可以在 CPU 和 GPU 之间共享的内存池。
    //    - 数据会自动按需在主机和设备之间迁移。
    //    - 简化编程，但可能带来性能开销，需要仔细管理以获得最佳性能
    //      (例如，使用 cudaMemAdvise 提示访问模式，使用 cudaStreamAttachMemAsync)。
    // =========================================================================
    std::cout << "--- 4. 统一内存 (Unified Memory) ---" << std::endl;
    {
        // 清空结果容器
        std::fill(h_data_result.begin(), h_data_result.end(), 0.0f);
        // 期望结果与第1部分相同 (doubleElements)

        float* um_data = nullptr; // 统一内存指针

        // 4.1 分配统一内存
        // cudaMallocManaged: 分配统一内存。
        // cudaMemAttachGlobal: 默认行为，内存可被任何 GPU 访问，按需迁移。
        // cudaMemAttachHost: 内存最初映射到主机，显式 GPU 访问时迁移。
        // cudaMemAttachSingle: 内存最初映射到特定 GPU。
        std::cout << "  分配统一内存 (um_data)... ";
        if (checkCuda(cudaMallocManaged(&um_data, dataSize, cudaMemAttachGlobal)) != cudaSuccess) return 1;
        std::cout << "完成." << std::endl;

        // 4.2 在 CPU 上初始化数据 (直接写入统一内存)
        std::cout << "  在 CPU 上初始化统一内存数据... ";
        for (int i = 0; i < N; ++i) {
            um_data[i] = h_data_original[i];
        }
        std::cout << "完成." << std::endl;

        // (可选) 提示数据位置
        // cudaMemPrefetchAsync: 可以异步地将统一内存区域预取到特定设备或主机。
        // 这可以帮助减少页面错误和迁移延迟。
        // int deviceId;
        // checkCuda(cudaGetDevice(&deviceId));
        // checkCuda(cudaMemPrefetchAsync(um_data, dataSize, deviceId, nullptr)); // nullptr for default stream

        // 4.3 在 GPU 上执行核函数 (直接操作统一内存)
        std::cout << "  启动 `doubleElements` 核函数 (操作统一内存)... ";
        doubleElements<<<blocksPerGrid, threadsPerBlock>>>(um_data, N);
        if (checkCuda(cudaGetLastError()) != cudaSuccess) { /* 清理 */ checkCuda(cudaFree(um_data)); return 1; }
        // 同步设备以确保核函数执行完成且数据迁移(如果需要)完成
        if (checkCuda(cudaDeviceSynchronize()) != cudaSuccess) { /* 清理 */ checkCuda(cudaFree(um_data)); return 1; }
        std::cout << "完成." << std::endl;
        
        // 4.4 在 CPU 上读取结果 (直接从统一内存读取)
        // 注意：在访问之前，如果数据在GPU上被修改，需要确保GPU操作已完成 (例如通过 cudaDeviceSynchronize)。
        //       或者，如果涉及流，需要同步该流。
        std::cout << "  在 CPU 上读取统一内存中的结果... ";
        for (int i = 0; i < N; ++i) {
            h_data_result[i] = um_data[i];
        }
        std::cout << "完成." << std::endl;
        
        // 4.5 验证结果
        if (std::equal(h_data_result.begin(), h_data_result.end(), h_data_expected.begin())) {
            std::cout << "  统一内存结果验证成功!" << std::endl;
        } else {
            std::cout << "  统一内存结果验证失败!" << std::endl;
        }

        // 4.6 释放统一内存
        // 对于统一内存，也使用 cudaFree。
        std::cout << "  释放统一内存 (um_data)... ";
        checkCuda(cudaFree(um_data));
        std::cout << "完成." << std::endl;
    }
    std::cout << std::endl;

    std::cout << "所有内存管理示例执行完毕." << std::endl;
    return 0;
} 