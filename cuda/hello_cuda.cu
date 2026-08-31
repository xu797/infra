#include <cstdio>

// __global__ 修饰符表明这是一个 GPU 上执行的 kernel 函数
__global__ void helloKernel() {
    // threadIdx.x 是当前线程在 Block 内的编号
    printf("Hello from GPU thread %d in block %d!\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    printf("Launching kernel...\n");

    // <<<2, 4>>> 表示启动 2 个 Block，每个 Block 有 4 个线程
    helloKernel<<<2, 100>>>();

    // 等待 GPU 上所有操作完成
    cudaDeviceSynchronize();

    printf("Done!\n");
    return 0;
}
