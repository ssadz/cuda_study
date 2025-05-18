 # 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言 (CXX 代表 C++, CUDA 代表 CUDA 语言)
project(HelloCUDA LANGUAGES CXX CUDA)

# 添加可执行文件目标
# add_executable(<目标名称> <源文件1> [源文件2 ...])
# 这里我们创建一个名为 hello_cuda 的可执行文件，源文件是 main.cu
add_executable(hello_cuda main.cu)

# 为目标设置属性
# CUDA_STANDARD 14: 指定 CUDA C++ 标准为 14
# CXX_STANDARD 14: 指定 C++ 标准为 14
# CXX_STANDARD_REQUIRED YES: 强制要求使用指定的 C++ 标准
set_target_properties(hello_cuda PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

# CMake 通常能自动识别 .cu 文件是 CUDA 语言。
# 如果遇到问题，可以取消下面这行的注释来显式指定：
# set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)

# 现代 CMake (3.8+ 启用 CUDA 语言支持后) 会自动处理 CUDA 运行时库的链接。
# 对于旧版本的 CMake 或者特定情况，可能需要手动链接，例如：
# find_package(CUDA REQUIRED)
# target_link_libraries(hello_cuda PRIVATE ${CUDA_LIBRARIES})
# 或者对于现代 CMake 的 target_link_libraries(hello_cuda PRIVATE CUDA::cudart)

# 输出一条状态消息，方便调试 CMake 过程
message(STATUS "配置 HelloCUDA 项目")