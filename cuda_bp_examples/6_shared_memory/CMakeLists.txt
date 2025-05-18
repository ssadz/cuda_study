# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(SharedMemoryExample LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(shared_memory_example shared_memory_main.cu)

# 为目标设置属性
set_target_properties(shared_memory_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

# 处理 MSVC 的 UTF-8 编译选项
if(MSVC)
    target_compile_options(shared_memory_example PRIVATE "-Xcompiler=/utf-8")
endif()

# 添加对cuBLAS库的链接
find_package(CUDAToolkit REQUIRED)
target_link_libraries(shared_memory_example PRIVATE CUDA::cublas)

message(STATUS "配置 SharedMemoryExample 项目 (位于 cuda_bp_examples/6_shared_memory)") 