`__shared__` 修饰，**属于同一个 thread block 的高速片上内存，在 GPU 芯片内部，不在显存 (global memory) 里**。


- Global Memory（全局显存，`gpu_input/gpu_output`）：慢，几百～上千个时钟周期，所有 block 都能访问。
- Shared Memory（共享内存）：**非常快，和寄存器接近**，几十 cycle；**仅同一个 block 内线程互相可见；block 之间完全隔离，A block 看不到 B block 的 shared memory**。