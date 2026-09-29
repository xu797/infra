#include<stdio.h>
#include<cuda_runtime.h>

#include<stdlib.h>
#include<time.h>
#include<math.h>

__global__ void sgemm(float *A_matrix, float *B_matrix, float *C_matrix, const int M, const int K, const int N)
{
    //global_memory...
    float *A_matrix_begin = A_matrix + blockDim.y * blockIdx.y * K;
    float *B_matrix_begin = B_matrix + blockDim.x * blockIdx.x;

    //first use reg...
    float sum = 0.0f;
    for(int i = 0; i < K; ++i)
    {
        sum += A_matrix_begin[i + threadIdx.y * K] * B_matrix_begin[i * N + threadIdx.x];
    }

    //copy data to c_matrix...
    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int x = blockDim.y * blockIdx.y + threadIdx.y;

    C_matrix[x * N + y] = sum;
}

bool check(float *cpu_res, float *gpu_res, int n)
{
    for(int i = 0; i < n; ++i)
    {   
        //int use abs(), float use fabsf()...
        if(fabsf(cpu_res[i] - gpu_res[i]) > 0.005f)
        {
            return false;
        }
    }

    return true;
}

// CPU sgemm 
void cpu_sgemm(const float* A, const float* B, float* C, const int M, const int K, const int N)
{
    // 先把C清零，你new()已经()初始化置0，保险起见再写一遍
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for(int t = 0; t < K; t++)
            {
                sum += A[i*K + t] * B[t*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void randnMatrix(float *matrix, int m, int n)
{
    int total = m * n;
    for(int i = 0; i < total; i++)
    {
        float r = rand() / (float)RAND_MAX; // [0,1]
        matrix[i] = 2.0f * r - 1.0f; // [-1, +1]
    }
}

int main()
{   
    /*
    A_matrix:M*K, B_matrix:K*N, C_matrix:M*N
    */
    const int M = 512;
    const int K = 256;
    const int N = 1024;

    //cpu malloc...
    float *cpuA_matrix = new float[M * K]();
    float *cpuB_matrix = new float[K * N]();
    float *cpuC_matrix = new float[M * N]();

    //cpu matrix init...
    srand((unsigned)time(nullptr));
    randnMatrix(cpuA_matrix, M, K);
    randnMatrix(cpuB_matrix, K, N);
    
    cpu_sgemm(cpuA_matrix, cpuB_matrix, cpuC_matrix, M, K, N);


    float *gpuA_matrix;
    float *gpuB_matrix;
    float *gpuC_matrix;

    //gpu malloc...
    cudaMalloc(&gpuA_matrix, M * K * sizeof(float));
    cudaMalloc(&gpuB_matrix, K * N * sizeof(float));
    cudaMalloc(&gpuC_matrix, M * N * sizeof(float));

    //gpu data init...
    cudaMemcpy(gpuA_matrix, cpuA_matrix, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB_matrix, cpuB_matrix, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gpuC_matrix, 0, M * N * sizeof(float));

    //kernel function...
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    sgemm<<<grid, block>>>(gpuA_matrix, gpuB_matrix, gpuC_matrix, M, K, N);

    //kernel error check...
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    //wait gpureslut...
    cudaDeviceSynchronize();

    //copy data from gpu to cpu...
    float *res = new float[M * N]();
    cudaMemcpy(res, gpuC_matrix, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    //check...
    if(check(cpuC_matrix, res, M * N))
    {
        printf("all right...\n");
    }
    else
    {
        printf("error...\n");
    }

    cudaFree(gpuA_matrix);
    cudaFree(gpuB_matrix);
    cudaFree(gpuC_matrix);

    delete []cpuA_matrix;
    delete []cpuB_matrix;
    delete []cpuC_matrix;
    delete []res;
}