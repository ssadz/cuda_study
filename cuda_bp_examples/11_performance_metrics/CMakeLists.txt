# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(PerformanceMetrics LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(performance_metrics performance_metrics.cu)

# 为目标设置属性
set_target_properties(performance_metrics PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

message(STATUS "配置 PerformanceMetrics 项目 (位于 cuda_bp_examples/11_performance_metrics)") 