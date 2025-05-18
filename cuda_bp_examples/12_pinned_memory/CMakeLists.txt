# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(PinnedMemoryExample LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(pinned_memory pinned_memory.cu)

# 为目标设置属性
set_target_properties(pinned_memory PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

message(STATUS "配置 PinnedMemoryExample 项目 (位于 cuda_bp_examples/12_pinned_memory)") 