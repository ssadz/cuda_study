# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义顶层项目名称和支持的语言
project(CUDAStudyProject LANGUAGES CXX CUDA)

# 添加 cuda_bp_examples 目录作为一个子目录
# CMake 会接着处理 cuda_bp_examples/CMakeLists.txt 文件
add_subdirectory(cuda_bp_examples)

# 输出一条状态消息（可选）
message(STATUS "配置 CUDAStudyProject (根项目)") 