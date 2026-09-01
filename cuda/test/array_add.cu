#include<stdio.h>
#include<cuda_runtime.h>

__global__ void array_add(int* arr, int* brr, int* crr, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n)
    {
        crr[i] = arr[i] + brr[i];
    }
}

int main()
{

    int n = 100000;
    int *cpu_arr = new int[n];
    int *cpu_brr = new int[n];
    int *cpu_crr = new int[n];
    for(int i = 0; i < n; ++i)
    {
        cpu_arr[i] = 1;
        cpu_brr[i] = 2;
    } 

    int *gpu_arr;
    int *gpu_brr;
    int *gpu_crr;
    cudaMalloc(&gpu_arr, n * sizeof(int));
    cudaMalloc(&gpu_brr, n * sizeof(int));
    cudaMalloc(&gpu_crr, n * sizeof(int));

    cudaMemcpy(gpu_arr, cpu_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_brr, cpu_brr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_crr, 0, n * sizeof(int));

    int thread = 256;
    int block = (n + thread - 1) / thread;
    array_add<<<block, thread>>>(gpu_arr, gpu_brr, gpu_crr, n);

    cudaMemcpy(cpu_crr, gpu_crr, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 100; ++i)
    {
        printf("the element of crr_cpu[%d]:%d\n", i, cpu_crr[i]);
    }

    cudaFree(gpu_arr);
    cudaFree(gpu_brr);
    cudaFree(gpu_crr);

    delete []cpu_arr;
    delete []cpu_brr;
    delete []cpu_crr;
}