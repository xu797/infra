#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(int *input, int *output)
{   

    //使用shuffle在warp内部的寄存器里面交换数据，比shared_memory速度更快
    int *input_begin = input + blockIdx.x * NUM_PER_BLOCK;
    int tid = threadIdx.x;
    int sum = 0;
    for(int i = 0; i < NUM_PER_BLOCK / THREAD_PER_BLOCK; ++i)
    {
        sum += input_begin[tid + i * THREAD_PER_BLOCK];
    }
    // 现在改成warp内规约，所以不需要同步操作
    // __syncthreads();
    
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    //最后所有warp算完之后，warp之间不能使用shuffle了，需要修改成shared_memory
    __shared__ int shared[32]; //理论上应该计算出来有多少个warp的 但是这里为了方便就直接写成32
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    
    if(laneId == 0)
    {
        shared[warpId] = sum;
    }

    __syncthreads();//这里需要同步了，因为要等所有warp都把shared_memory写完之后才能往下走

    //然后使用第一个warp来完成最后的计算，把shared_memory放到warp0里面
    if(warpId == 0)
    {
        //虽然开了32个shared,但是后面可能有一部分没有用到，也就是wrap数量小于32
        sum = (laneId < blockDim.x / 32) ? shared[laneId] : 0;

        // sum = shared[laneId]; //warp0内部所有thread拿完之后，才会往下走

        //这里不需要同步，现在是在warp0内部，thread同步
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if(threadIdx.x == 0)
    {
        output[blockIdx.x] = sum;
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