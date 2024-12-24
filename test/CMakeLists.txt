# 设置测试目标
add_executable(cuda_pixel_error_calc_test test.cpp)

# 链接静态库
target_link_libraries(cuda_pixel_error_calc_test PRIVATE cuda_pixel_error_calc)

# 添加包含路径（如果需要访问头文件）
target_include_directories(cuda_pixel_error_calc_test PRIVATE ${PROJECT_SOURCE_DIR}/ext ${PROJECT_SOURCE_DIR}/include) 
