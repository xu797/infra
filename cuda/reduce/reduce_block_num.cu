#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(int *input, int *output)
{   
    //volatile就是在最后一个warp里面每次强制刷新shared_memory 确保每一次操作都写回去了 而不是写到了寄存器里面
    volatile __shared__ int shared[THREAD_PER_BLOCK];

    int *input_begin = input + blockIdx.x * NUM_PER_BLOCK;
    int tid = threadIdx.x;
    shared[tid] = 0;
    for(int i = 0; i < NUM_PER_BLOCK / THREAD_PER_BLOCK; ++i)
    {
        shared[tid] += input_begin[tid + i * THREAD_PER_BLOCK];
    }
    __syncthreads();

    for(int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if(threadIdx.x < i)
        {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads(); //block里面所有thread执行完之后才能往下走.
    }

    if(threadIdx.x < 32)
    {
        shared[threadIdx.x]+=shared[threadIdx.x+32];
        shared[threadIdx.x]+=shared[threadIdx.x+16];
        shared[threadIdx.x]+=shared[threadIdx.x+8];
        shared[threadIdx.x]+=shared[threadIdx.x+4];
        shared[threadIdx.x]+=shared[threadIdx.x+2];
        shared[threadIdx.x]+=shared[threadIdx.x+1];
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
    constexpr int N = 3 * 1024 * 1024;
    constexpr int BLOCK_NUM = 512;
    // 一个block需要处理的数量
    constexpr int num_per_block = N / BLOCK_NUM; // ✅编译期计算6144
    

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
        for(int j = 0; j < num_per_block; ++j)
        {
            cpu_output[i] += cpu_input[j + i * num_per_block];
        }
    }


    int *gpu_input;
    int *gpu_output;
    cudaMalloc(&gpu_input, N * sizeof(int));
    cudaMalloc(&gpu_output, BLOCK_NUM * sizeof(int));

    cudaMemcpy(gpu_input, cpu_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_output, 0, BLOCK_NUM * sizeof(int));

    reduce<num_per_block><<<BLOCK_NUM, THREAD_PER_BLOCK>>>(gpu_input, gpu_output);

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