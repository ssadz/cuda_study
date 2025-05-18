# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义项目名称和支持的语言
project(InstructionOptimizationsExample LANGUAGES CXX CUDA)

# 拆分为四个子示例，分别演示不同指令级优化技术
add_executable(std_math_example standard_math.cu)
set_target_properties(std_math_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

add_executable(intrinsic_math_example intrinsic_math.cu)
set_target_properties(intrinsic_math_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

add_executable(double_precision_example double_precision.cu)
set_target_properties(double_precision_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

add_executable(fma_example fma_example.cu)
set_target_properties(fma_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

option(USE_FAST_MATH_FOR_INSTR_OPT "Enable -use_fast_math for all instruction optimization examples" ON)
if(USE_FAST_MATH_FOR_INSTR_OPT)
  target_compile_options(std_math_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
  target_compile_options(intrinsic_math_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
  target_compile_options(double_precision_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
  target_compile_options(fma_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
  message(STATUS "Fast math enabled for all instruction optimization examples")
endif()

# Warp Shuffle 示例: 演示 warp 级别的 __shfl_sync 用法
add_executable(warp_shuffle_example warp_shuffle_example.cu)
set_target_properties(warp_shuffle_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
if(USE_FAST_MATH_FOR_INSTR_OPT)
  target_compile_options(warp_shuffle_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()
message(STATUS "配置 WarpShuffleExample 项目 (位于 cuda_bp_examples/5_instruction_optimizations)")

# 为兼容旧目标名，添加 alias 目标
add_custom_target(instr_opt_example
    DEPENDS std_math_example intrinsic_math_example double_precision_example fma_example
)
set_property(TARGET instr_opt_example APPEND PROPERTY DEPENDS warp_shuffle_example)

# Vectorization 示例: 演示使用 float4 一次加载多元素提高带宽
add_executable(vectorization_example vectorization_example.cu)
set_target_properties(vectorization_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
if(USE_FAST_MATH_FOR_INSTR_OPT)
  target_compile_options(vectorization_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()
message(STATUS "配置 VectorizationExample 项目 (位于 cuda_bp_examples/5_instruction_optimizations)")

# 为兼容旧目标名，将 vectorization_example 添加到 instr_opt_example alias
set_property(TARGET instr_opt_example APPEND PROPERTY DEPENDS vectorization_example)

# Read-only cache 示例: 使用 __ldg 访问只读数据缓存
add_executable(readonly_cache_example readonly_cache_example.cu)
set_target_properties(readonly_cache_example PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
if(USE_FAST_MATH_FOR_INSTR_OPT)
  target_compile_options(readonly_cache_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()
message(STATUS "配置 ReadOnlyCacheExample 项目 (位于 cuda_bp_examples/5_instruction_optimizations)")

# 将 readonly_cache_example 添加到 instr_opt_example alias
set_property(TARGET instr_opt_example APPEND PROPERTY DEPENDS readonly_cache_example)

message(STATUS "配置 InstructionOptimizationsExample 项目 (位于 cuda_bp_examples/5_instruction_optimizations)") 