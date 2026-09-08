#include<stdio.h>
#include<cuda_runtime.h>

#include<stdlib.h>
#include<time.h>
#include<math.h>

#define FETCH_FLOAT4(pointer) (*((float4*)(&(pointer))))


template<unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, 
unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void sgemm(float *A_matrix, float *B_matrix, float *C_matrix, const int M, const int K, const int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *A_matrix_begin = A_matrix + K * blockIdx.y * M_NUM_PER_BLOCK;
    float *B_matrix_begin = B_matrix + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float A_matrix_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float B_matrix_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD] = {0.0f};

    for(int i = 0; i < K; i += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(A_matrix_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_matrix_begin[K * ty + i + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(B_matrix_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_matrix_begin[(ty + i) * N + tx * NUM_PER_THREAD]);
        __syncthreads();
        
        for(int j = 0; j < NUM_PER_THREAD; ++j)
        {
            for(int k = 0; k < K_NUM_PER_BLOCK; ++k)
            {
                temp[j] += A_matrix_shared[ty][k] * B_matrix_shared[k][tx * NUM_PER_THREAD + j];
            }
        }
        __syncthreads();
    }

    float *C_matrix_begin = C_matrix + N * blockIdx.y * M_NUM_PER_BLOCK + blockIdx.x * N_NUM_PER_BLOCK;

    for(int i = 0; i < NUM_PER_THREAD; ++i)
    {
        C_matrix_begin[ty * N + tx * NUM_PER_THREAD + i] = temp[i];
    }

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
    const int K = 512;
    const int N = 512;

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
    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    dim3 block(8,32);
    dim3 grid(M / M_NUM_PER_BLOCK, N / N_NUM_PER_BLOCK);

    sgemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(gpuA_matrix, gpuB_matrix, gpuC_matrix, M, K, N);

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