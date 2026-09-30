#include<iostream>
#include<cuda_runtime.h>

__global__ void softmax(float *input, float *output, int M, int N)
{   
    //处理index行
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > M)
    {
        return;
    }

    float *input_begin = input + index * N;
    float *output_begin = output + index * N;

    float pre_max_value = -INFINITY;
    float max_value = -INFINITY;
    float sum = 0.0f;

    for(int i = 0; i < N; ++i)
    {
        max_value = fmaxf(input_begin[i], pre_max_value);
        sum = sum * expf(pre_max_value - max_value) + expf(input_begin[i] - max_value);
        pre_max_value = max_value;
    }

    float inv_sum = 1.0f / sum;
    for(int i = 0; i < N; ++i)
    {
        output_begin[i] = expf(input_begin[i] - max_value) * inv_sum;
    }

}

void softmax_cpu(float *input, float *output, int M, int N)
{
    //M 行
    for(int i = 0; i < M; ++i)
    {      
        //max_value
        float max_value = -INFINITY;
        for(int j = 0; j < N; ++j)
        {
            max_value = std::max(max_value, input[i * N + j]);
        }
        //get sum
        float sum = 0.0f;
        for(int j = 0; j < N; ++j)
        {
            sum += std::exp(input[i * N + j] - max_value);
        }
        //write output
        for(int j = 0; j < N; ++j)
        {
            output[i * N + j] = input[i * N + j] / sum;
        }
    }
}

void check(float *res1, float *res2, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {   
        for(int j = 0; j < N; ++j)
        {
            //int use abs(), float use fabsf()...
            if(fabsf(res1[i * N + j] - res2[i * N + j]) > 0.005f)
            {
                std::cout << "the result is error..." << std::endl;
            }
        }
    }
    std::cout << "the result is right..." << std::endl;
}

int main()
{
    const int M = 512;
    const int N = 1024;
    float *input_cpu = new float[M * N]();
    float *output_cpu = new float[M * N]();
    softmax_cpu(input_cpu, output_cpu, M, N);

    float *input_gpu;
    float *output_gpu;

    cudaMalloc(&input_gpu, sizeof(float) * M * N);
    cudaMalloc(&output_gpu, sizeof(float) * M * N);

    cudaMemcpy(input_gpu, input_cpu, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemset(output_gpu, 0, sizeof(float) * M * N);

    int block_size = 128;
    int grid_size = (M + block_size - 1) / block_size;
    softmax<<<grid_size, block_size>>>(input_gpu, output_gpu, M, N);

    //wait kernel to process...
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float *res = new float[M * N]();
    cudaMemcpy(res, output_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    check(res, output_cpu, M, N);


}