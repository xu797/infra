find_package(CUDAToolkit REQUIRED)

add_executable(flash_atten_v1 flash_atten_v1.cu)
target_link_libraries(flash_atten_v1 PRIVATE CUDA::cudart)


