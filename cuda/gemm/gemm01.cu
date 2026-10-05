#include<iostream>
#include<cmath>
#include <iomanip>
#include<cuda_runtime_api.h>

__global__ void sgemm_naive(float *A_matrix, float *B_matrix, float *Output_matrix, int M, int K, int N)
{
    //每个thread计算out_matrix的一个值
    int tx = blockIdx.x * blockDim.x + threadIdx.x; //lie
    int ty = blockIdx.y * blockDim.y + threadIdx.y; //hang

    if (ty >= M || tx >= N)
    {
        return ;
    }

    float sum = 0.0f;
    for(int i = 0; i < K; ++i)
    {
        sum += A_matrix[ty * K + i] * B_matrix[i * N + tx];
    }

    Output_matrix[ty * N + tx] = sum;
}

bool check(float *res1, float *res2, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {   
        for(int j = 0; j < N; ++j)
        {
            //int use abs(), float use fabsf()...
            if(fabsf(res1[i * N + j] - res2[i * N + j]) > 0.005f)
            {
                // std::cout << "the result is error..." << std::endl;
                return false;
            }
        }
    }
    // std::cout << "the result is right..." << std::endl;
    return true;
}

void view_result(float *res_cpu, float *res_gpu, int M, int N)
{
    // std::cout << "left: cpu_result, right: gpu_result" << std::endl;
    // for(int i = 0; i < 20; ++i)
    // {
    //     std::cout << res_cpu[i] << "    " << res_gpu[i] << std::endl;
    // }
    std::cout << std::fixed << std::setprecision(8);
    std::cout << std::left
              << std::setw(18) << "left: cpu_result"
              << std::setw(18) << "right: gpu_result"
              << std::endl;

    for(int i = 0; i < M; ++i)
    {
        std::cout << std::setw(18) << res_cpu[i * N]
                  << std::setw(18) << res_gpu[i * N]
                  << std::endl;
    }
}


void cpu_result(float *A_matrix, float *B_matrix, float *Output_matrix, int M, int K, int N)
{
    float sum = 0.0f;
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            for(int k = 0; k < K; ++k)
            {
                sum += A_matrix[i * K + k] * B_matrix[j + k * N]; 
            }
            Output_matrix[i * N + j] = sum;
            sum = 0.0f;
        }
    }
}

int main()
{
    const int M = 512;
    const int K = 1024;
    const int N = 256;

    float *A_matrix_cpu = new float[M * K]();
    float *B_matrix_cpu = new float[K * N]();
    float *Output_matrix_cpu = new float[M * N]();

    for(int i = 0; i < M * K; i++)
    {
        A_matrix_cpu[i] = rand() * 1.0f / RAND_MAX;
    }

    for(int i = 0; i < K * N; i++)
    {
        B_matrix_cpu[i] = rand() * 1.0f / RAND_MAX;
    }

    cpu_result(A_matrix_cpu, B_matrix_cpu, Output_matrix_cpu, M, K, N);

    float *A_matrix_gpu;
    float *B_matrix_gpu;
    float *Output_matrix_gpu;

    cudaMalloc(&A_matrix_gpu, sizeof(float) * M * K);
    cudaMalloc(&B_matrix_gpu, sizeof(float) * K * N);
    cudaMalloc(&Output_matrix_gpu, sizeof(float) * M * N);

    cudaMemcpy(A_matrix_gpu, A_matrix_cpu, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_matrix_gpu, B_matrix_cpu, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemset(Output_matrix_gpu, 0, sizeof(float) * M * N);

    dim3 block(16,16);
    dim3 grid((N + block.x -1)/block.x, (M + block.y -1)/block.y);

    sgemm_naive<<<grid, block>>>(A_matrix_gpu, B_matrix_gpu, Output_matrix_gpu, M, K, N);
        
    //wait kernel to process...
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float *res = new float[M * N]();
    cudaMemcpy(res, Output_matrix_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    view_result(Output_matrix_cpu, res, M, N);

    if(check(res, Output_matrix_cpu, M, N))
    {
        std::cout << "the result is right..." << std::endl;
    }
    else
    {
        std::cout << "the result is error..." << std::endl;
    }

    delete []A_matrix_cpu;
    delete []B_matrix_cpu;
    delete []Output_matrix_cpu;
    delete []res;

    cudaFree(A_matrix_gpu);
    cudaFree(B_matrix_gpu);
    cudaFree(Output_matrix_gpu);
}