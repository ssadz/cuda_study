# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(ExecutionConfigurationExample LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(exec_config_example execution_main.cu)

# 为目标设置属性
set_target_properties(exec_config_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

if(MSVC)
    # 将 /utf-8 传递给主机编译器 (MSVC)
    target_compile_options(exec_config_example PRIVATE "-Xcompiler=/utf-8")
endif()

message(STATUS "配置 ExecutionConfigurationExample 项目 (位于 cuda_bp_examples/4_execution_configuration)") 