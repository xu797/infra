find_package(CUDAToolkit REQUIRED)

add_executable(softmax_v1 softmax_v1.cu)
target_link_libraries(softmax_v1 PRIVATE CUDA::cudart)

add_executable(softmax_v2 softmax_v2.cu)
target_link_libraries(softmax_v2 PRIVATE CUDA::cudart)



