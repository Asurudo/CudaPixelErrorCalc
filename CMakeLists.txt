# cmake 最低版本
cmake_minimum_required(VERSION 3.15)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 项目名称
project(cuda_pixel_error_calc LANGUAGES CUDA CXX)

# 设置静态库输出地址
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 生成静态库
add_library(cuda_pixel_error_calc STATIC ${PROJECT_SOURCE_DIR}/src/CPEC.cu)

set_target_properties(cuda_pixel_error_calc PROPERTIES COMPILE_OPTIONS "-Xcompiler;/utf-8")

target_include_directories(cuda_pixel_error_calc PRIVATE  ${PROJECT_SOURCE_DIR}/ext/stb/  ${PROJECT_SOURCE_DIR}/include)

# 添加 test 子目录
add_subdirectory(test)