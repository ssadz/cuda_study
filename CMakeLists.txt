# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 定义顶层项目名称和支持的语言
project(CUDAStudyProject LANGUAGES CXX CUDA)

# 设置 C++ 标准为 C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# 为 CUDA 项目启用可重定位设备代码 (RDC)
# 这对于跨多个编译单元的设备代码链接是必要的
# 只使用最常见的CUDA架构，避免前向兼容性问题
set(CMAKE_CUDA_ARCHITECTURES "60") # Pascal架构，较为通用

# 注意：如果不需要跨多个CUDA源文件链接，可以禁用RDC以提高性能

# 如果特定项目需要RDC，可以在其CMakeLists.txt中单独设置
# 例如: set_target_properties(target_name PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 使用现代CMake方式处理CUDA，不再使用已弃用的FindCUDA模块
# 设置策略避免警告
cmake_policy(SET CMP0146 NEW)

# 输出一些有用的 CUDA 变量 (可选，用于调试)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA Include Dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    
    message(STATUS "\n注意：如果编译成功但无法运行CUDA程序，请检查：")
    message(STATUS "1. 是否正确安装了NVIDIA驱动")
    message(STATUS "2. 运行nvidia-smi检查驱动状态 - 若有版本不匹配，请更新驱动")
    message(STATUS "3. 查看'lspci | grep -i nvidia'确认系统检测到NVIDIA设备")
    message(STATUS "4. 尝试'sudo apt update && sudo apt install nvidia-driver-<版本>' 更新驱动\n")
else()
    message(FATAL_ERROR "CUDA compiler not found!")
endif()

# 添加 cuda_bp_examples 目录作为一个子目录
# CMake 会接着处理 cuda_bp_examples/CMakeLists.txt 文件
add_subdirectory(cuda_bp_examples)

# 输出一条状态消息（可选）
message(STATUS "配置 CUDAStudyProject (根项目)")

# 如果您计划在顶层项目直接添加源文件，可以在这里添加可执行文件目标
# 例如:
# file(GLOB_RECURSE MAIN_SOURCES src/*.cpp src/*.cu)
# if(MAIN_SOURCES)
#     add_executable(MainApp ${MAIN_SOURCES})
#     target_link_libraries(MainApp PRIVATE ${CUDA_LIBRARIES})
#     # 对于 CUDA，可能还需要链接 cudart_static 或 cudart
#     if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.8")
#         target_link_libraries(MainApp PRIVATE CUDA::cudart)
#     else()
#         target_link_libraries(MainApp PRIVATE ${CUDA_LIBRARIES})
#     endif()
#     message(STATUS "配置顶层可执行文件 MainApp")
# else()
#     message(STATUS "顶层项目未发现 src 目录或源文件，未创建 MainApp 可执行文件。")
# endif() 