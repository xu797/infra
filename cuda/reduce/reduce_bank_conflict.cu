#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce(int *input, int *output)
{   
    __shared__ int shared[THREAD_PER_BLOCK];

    int *input_begin = input + blockIdx.x * blockDim.x;

    // 要等一个block里面的所有thread完成共享内存搬运之后才往下走
    shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    for(int i = blockDim.x; i > 0; i /= 2)
    {
        if(threadIdx.x < i)
        {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads(); //block里面所有thread执行完之后才能往下走.
    }


    if(threadIdx.x == 0)
    {
        output[blockIdx.x] = shared[0];
    }
}

bool check(int *arr, int *brr, int n)
{
    for(int i = 0; i < n; ++i)
    {
        if(arr[i] != brr[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{   
    int N = 3 * 1024 * 1024;
    int BLOCK_NUM = (N + 255) / THREAD_PER_BLOCK;
    int *cpu_input = new int[N];
    int *cpu_output = new int[BLOCK_NUM];

    srand((unsigned)time(nullptr));
    for(int i = 0; i < N; ++i)
    {
        cpu_input[i] = rand() % 50;
       
    }
    for(int i = 0; i < BLOCK_NUM; ++i)
    {
        cpu_output[i] = 0;
    }

    // cpu_output
    for(int i = 0; i < BLOCK_NUM; ++i)
    {
        for(int j = 0; j < THREAD_PER_BLOCK; ++j)
        {
            cpu_output[i] += cpu_input[j + i * THREAD_PER_BLOCK];
        }
    }


    int *gpu_input;
    int *gpu_output;
    cudaMalloc(&gpu_input, N * sizeof(int));
    cudaMalloc(&gpu_output, BLOCK_NUM * sizeof(int));

    cudaMemcpy(gpu_input, cpu_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_output, 0, BLOCK_NUM * sizeof(int));

    reduce<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(gpu_input, gpu_output);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    int *res = new int[BLOCK_NUM];
    cudaMemcpy(res, gpu_output, BLOCK_NUM * sizeof(int), cudaMemcpyDeviceToHost);
    if(check(res, cpu_output, BLOCK_NUM))
    {
        printf("all right...\n");
    }else{
        printf("error...\n");
    }

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    delete []cpu_input;
    delete []cpu_output;
    delete []res;

}