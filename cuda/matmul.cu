#include<stdio.h>
#include<cuda_runtime.h>

__global__ void matmul_add(float *arr, float *brr, float *crr, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if(i < width && j < height)
    {
        int index = j * width + i;
        crr[index] = brr[index] + arr[index];
    }

}



int main()
{
    int width = 1024, height = 1024;
    int size = width * height;
    float *cpu_arr = new float[size];
    float *cpu_brr = new float[size];
    float *cpu_crr = new float[size];

    for(int i = 0; i < size; ++i)
    {
        cpu_arr[i] = 1.0;
        cpu_brr[i] = 2.0;
        cpu_crr[i] = 0.0;
    }
    
    float *gpu_arr;
    float *gpu_brr;
    float *gpu_crr;

    cudaMalloc(&gpu_arr, size * sizeof(float));
    cudaMalloc(&gpu_brr, size * sizeof(float));
    cudaMalloc(&gpu_crr, size * sizeof(float));

    cudaMemcpy(gpu_arr, cpu_arr, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_brr, cpu_brr, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gpu_crr, 0, size * sizeof(float));

    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) /16);

    matmul_add<<<grid_size, block_size>>>(gpu_arr, gpu_brr, gpu_crr, width, height);

    //检查kernel错误
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(cpu_crr, gpu_crr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < size; ++i)
    // {
    //     printf("%f ", cpu_crr[i]);
    //     if((i + 1) % width ==0)
    //     {
    //         printf("\n");
    //     }
    // }

    cudaFree(gpu_arr);
    cudaFree(gpu_brr);
    cudaFree(gpu_crr);

    delete []cpu_arr;
    delete []cpu_brr;
    delete []cpu_crr;


}