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

add_executable(reduce_idle_thread_A reduce_idle_thread_A.cu)
target_link_libraries(reduce_idle_thread_A PRIVATE CUDA::cudart)

add_executable(reduce_idle_thread_B reduce_idle_thread_B.cu)
target_link_libraries(reduce_idle_thread_B PRIVATE CUDA::cudart)

add_executable(reduce_unroll_last_warp reduce_unroll_last_warp.cu)
target_link_libraries(reduce_unroll_last_warp PRIVATE CUDA::cudart)

add_executable(reduce_block_num reduce_block_num.cu)
target_link_libraries(reduce_block_num PRIVATE CUDA::cudart)

add_executable(reduce_shuffle reduce_shuffle.cu)
target_link_libraries(reduce_shuffle PRIVATE CUDA::cudart)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
# set_target_properties(reduce_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)