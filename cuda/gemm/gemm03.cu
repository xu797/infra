#include<iostream>
#include<cmath>
#include <iomanip>
#include<cuda_runtime_api.h>

template<int BM, int BK, int BN, int BLOCK_SIZE>
__global__ void sgemm02(float *A_matrix, float *B_matrix, float *Output_matrix, int M, int K, int N)
{
    //blcok_size:256
    //BM:128, BK:8，BN:128
    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    //matrix_begin
    float *A_begin = A_matrix + BM * blockIdx.y * K;
    float *B_begin = B_matrix + BN * blockIdx.x;

    //线程重排
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X; //256个thread: [32, 8], a_thread_y = [0...32]

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X; //256个thread: [8, 32], a_thread_y = [0...8]

    // for c_matrix:[128, 128], block_size:[16, 16]
    //一个线程负责x方向128/16个 y方向128/16个
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X;
    int c_thread_y = tid / C_BLOCK_X;

    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8
    constexpr int Tn = BN / C_BLOCK_X;  // = 8
    float Ct[Tm][Tn] = {0.0f};

    //reg_cache
    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};`

    //大循环次数: K / bk,一次大循环计算出来的是一个block的结果：out[BM][BN]
    for(int s = 0; s < K; s += BK)
    {
        // load A_matrix
        for(int i = a_thread_y; i < BM; i += A_BLOCK_Y)
        {
            A_shared[i][a_thread_x] = A_begin[i * K + a_thread_x + s];
        }

        //load B_matrix
        for(int i = b_thread_x; i < BN; i += B_BLOCK_X)
        {
            B_shared[b_thread_y][i] = B_begin[(b_thread_y + s) * N + i];
        }

        __syncthreads();
        //output_matrix
        //取出A_shared(列):Tm个元素，B_shared(行): tn个元素
        for(int i = 0; i < BK; ++i)
        {
            for(int j = 0; j < Tm; ++j)
            {
                a_frag[j] = A_shared[c_thread_y * Tm + j][i];
            }
            for(int j = 0; j < Tn; ++j)
            {
                b_frag[j] = B_shared[i][c_thread_x * Tn + j];
            }
            //外积
            for(int j = 0; j < Tm; ++j)
            {
                for(int k = 0; k < Tn; ++k)
                {
                    Ct[j][k] += a_frag[j] * b_frag[k]; 
                }
            }
        }

        __syncthreads();
    }

    //write to output_matrix
    float *C_begin = Output_matrix + blockIdx.y * BM * N + blockIdx.x * BN;

    //c_begin_size:[BM, BN]

    for(int i = 0; i < Tm; ++i)
    {
        for(int j = 0; j < Tn; ++j)
        {
            C_begin[(c_thread_y * Tm + i) * N + (c_thread_x * Tn + j)] = Ct[i][j];
        }
    }

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

    const int BM = 128;
    const int BK = 8;
    const int BN = 128;
    const int BLOCK_SIZE = 256;
    // dim3 block(16,16);
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm02<BM, BK, BN, BLOCK_SIZE><<<grid, block>>>(A_matrix_gpu, B_matrix_gpu, Output_matrix_gpu, M, K, N);
        
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