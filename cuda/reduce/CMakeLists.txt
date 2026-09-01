#find cuda
find_package(CUDAToolkit)

#add_executable(hello_reduce hello_reduce.cu)

# target_link_libraries(hello_reduce PRIVATE CUDA::cudart)
# add_executable(reduce reduce.cu)
# target_link_libraries(reduce PRIVATE CUDA::cudart)

add_executable(reduce_shared_memory reduce_shared_memory.cu)
target_link_libraries(reduce_shared_memory PRIVATE CUDA::cudart)

add_executable(reduce_warp_divergence reduce_warp_divergence.cu)
target_link_libraries(reduce_warp_divergence PRIVATE CUDA::cudart)

add_executable(reduce_bank_conflict reduce_bank_conflict.cu)
target_link_libraries(reduce_bank_conflict PRIVATE CUDA::cudart)