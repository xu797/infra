find_package(CUDAToolkit REQUIRED)

add_executable(attention attention.cu)
target_link_libraries(attention PRIVATE CUDA::cudart)

add_executable(flash_atten_v1 flash_atten_v1.cu)
target_link_libraries(flash_atten_v1 PRIVATE CUDA::cudart)


