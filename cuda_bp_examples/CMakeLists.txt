# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义顶层项目名称和支持的语言
project(CUDABestPracticesExamples LANGUAGES CXX CUDA)

# 全局设置 C++ 和 CUDA 标准 (这些设置会影响所有子目录)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES) # 确保严格使用 C++14
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED YES) # 确保严格使用 CUDA C++14

# 添加子目录
# 这会告诉 CMake 去相应的文件夹中查找并处理其 CMakeLists.txt 文件
add_subdirectory(0_hello_cuda)
# 添加新的 APOD 示例子目录
add_subdirectory(2_apod_example)
add_subdirectory(3_memory_management)
add_subdirectory(4_execution_configuration)
add_subdirectory(5_instruction_optimizations)
add_subdirectory(6_shared_memory)
add_subdirectory(7_constant_memory)
add_subdirectory(11_performance_metrics)
add_subdirectory(12_pinned_memory)
add_subdirectory(13_execution_configuration)
add_subdirectory(15_control_flow)
add_subdirectory(8_texture_memory)

# 输出一条状态消息
message(STATUS "配置 CUDABestPracticesExamples 根项目 (位于 cuda_bp_examples)")