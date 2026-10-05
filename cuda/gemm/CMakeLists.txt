find_package(CUDAToolkit REQUIRED)

add_executable(gemm01 gemm01.cu)
target_link_libraries(gemm01 PRIVATE CUDA::cudart)

add_executable(gemm02 gemm02.cu)
target_link_libraries(gemm02 PRIVATE CUDA::cudart)
