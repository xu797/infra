#include<cstdio>

__global__ void array_sum(int *arr, int *res, int n)
{
    int i = threadIdx.x;
    if(i < n)
    {
        atomicAdd(res, arr[i]);
    }
}

int main()
{
    int n = 100;
    int* cpu_arr = new int[n];
    int cpu_res;
    for(int i = 0; i < n; ++i)
    {
        cpu_arr[i] = i; 
    }
    int* gpu_arr;
    int* gpu_res;
    //gpu分配内存
    cudaMalloc(&gpu_arr, n * sizeof(int));
    cudaMalloc(&gpu_res, sizeof(int));

    //拷贝数据到gpu
    cudaMemcpy(gpu_arr, cpu_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_res, 0, sizeof(int));

    array_sum<<<1, n>>>(gpu_arr, gpu_res, n);

    cudaMemcpy(&cpu_res, gpu_res, sizeof(int), cudaMemcpyDeviceToHost);
    printf("sum of cpu_arr:%d\n", cpu_res);

    cudaFree(gpu_arr);
    cudaFree(gpu_res);

    delete []cpu_arr;

}