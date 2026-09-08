find_package(CUDAToolkit REQUIRED)

add_executable(hello_sgemm hello_sgemm.cu)
target_link_libraries(hello_sgemm PRIVATE CUDA::cudart)

add_executable(sgemm01_baseline sgemm01_baseline.cu)
target_link_libraries(sgemm01_baseline PRIVATE CUDA::cudart)

add_executable(sgemm02_shared_memory sgemm02_shared_memory.cu)
target_link_libraries(sgemm02_shared_memory PRIVATE CUDA::cudart)

add_executable(sgemm03_shared_memory sgemm03_shared_memory.cu)
target_link_libraries(sgemm03_shared_memory PRIVATE CUDA::cudart)

add_executable(sgemm04_per_thread_work sgemm04_per_thread_work.cu)
target_link_libraries(sgemm04_per_thread_work PRIVATE CUDA::cudart)

add_executable(sgemm05_float4 sgemm05_float4.cu)
target_link_libraries(sgemm05_float4 PRIVATE CUDA::cudart)
