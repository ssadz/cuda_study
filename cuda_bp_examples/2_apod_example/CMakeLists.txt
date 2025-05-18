# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(APODExample LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(apod_example apod_main.cu)

# 为目标设置属性
# CUDA_STANDARD 14: 指定 CUDA C++ 标准为 14
# CXX_STANDARD 14: 指定 C++ 标准为 14
# CXX_STANDARD_REQUIRED YES: 强制要求使用指定的 C++ 标准
set_target_properties(apod_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

message(STATUS "配置 APODExample 项目 (位于 cuda_bp_examples/apod_example)") 