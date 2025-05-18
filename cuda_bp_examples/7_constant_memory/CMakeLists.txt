# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(ConstantMemoryExample LANGUAGES CXX CUDA)

# 添加可执行文件目标
add_executable(constant_memory_example constant_memory_example.cu)

# 设置目标属性：C++和CUDA标准
set_target_properties(constant_memory_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

# 可选：添加编译选项
# 例如，可以启用fast-math来提高性能（但可能会影响精度）
# target_compile_options(constant_memory_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

# 输出状态消息
message(STATUS "配置 ConstantMemoryExample 项目 (位于 cuda_bp_examples/7_constant_memory)") 