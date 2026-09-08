#include<stdio.h>
#include<cuda_runtime.h>

#include<stdlib.h>
#include<time.h>
#include<math.h>



template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void sgemm(float *A_matrix, float *B_matrix, float *C_matrix, const int M, const int K, const int N)
{
    //shared_memory...
    const int STEP = BLOCK_SIZE * STRIDE;

    float *A_matrix_begin = A_matrix + STEP * blockIdx.y * K;
    float *B_matrix_begin = B_matrix + STEP * blockIdx.x;

    __shared__ float A_matrix_shared[STEP][STEP];
    __shared__ float B_matrix_shared[STEP][STEP];

    float sum[STRIDE][STRIDE] = {0.0f};

    for(int i = 0; i < K; i += STEP)
    {
        for(int j = 0; j < STRIDE; ++j)
        {
            for(int k = 0; k < STRIDE; ++k)
            {
                A_matrix_shared[threadIdx.y + j * BLOCK_SIZE][threadIdx.x + k * BLOCK_SIZE] = A_matrix_begin[(threadIdx.y + j * BLOCK_SIZE) * K + threadIdx.x + k * BLOCK_SIZE + i];
                B_matrix_shared[threadIdx.y + j * BLOCK_SIZE][threadIdx.x + k * BLOCK_SIZE] = B_matrix_begin[(threadIdx.y + j * BLOCK_SIZE + i) * N + threadIdx.x + k * BLOCK_SIZE];
            }
        }
        //同步，计算小block
        __syncthreads();

        for(int j = 0; j < STRIDE; ++j)
        {
            for(int k = 0; k < STRIDE; ++k)
            {
                for(int t = 0; t < STEP; ++t)
                {
                    sum[j][k] += A_matrix_shared[threadIdx.y + j * BLOCK_SIZE][t] * B_matrix_shared[t][threadIdx.x + k * BLOCK_SIZE];
                }   
            }
        }
        //同步，等shared内存使用完之后，才进行下一个循环
        __syncthreads();
    }
    
    //copy data to c_matrix...
    float * C_matrix_begin = C_matrix + N * blockIdx.y * STEP + blockIdx.x * STEP;
    for(int j = 0; j < STRIDE; ++j)
        {
            for(int k = 0; k < STRIDE; ++k)
            {
                C_matrix_begin[(threadIdx.y + j * BLOCK_SIZE) * N + threadIdx.x + k * BLOCK_SIZE] = sum[j][k];
            }
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
    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK / 2, (N + BLOCK - 1) / BLOCK / 2);
    sgemm<BLOCK, STRIDE><<<grid, block>>>(gpuA_matrix, gpuB_matrix, gpuC_matrix, M, K, N);

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