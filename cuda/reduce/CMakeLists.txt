#find cuda
find_package(CUDAToolkit)

#add_executable(hello_reduce hello_reduce.cu)

# target_link_libraries(hello_reduce PRIVATE CUDA::cudart)
add_executable(reduce reduce.cu)
target_link_libraries(reduce PRIVATE CUDA::cudart)