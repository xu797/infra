#include <iostream>
#include<cmath>
#include <iomanip>
#include <cuda_runtime_api.h>

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

void attention_cpu(float *q_matrix, float *k_matrix, float *v_matrix, float *output_matrix, int m, int n)
{
    float *s_matrix = new float[m * m]();
    //q*k^t
    for(int i = 0; i < m; ++i)
    {
        float *q_begin = q_matrix + i * n;
        float *s_begin = s_matrix + i * m;
        float sum = 0.0f;
        for(int j = 0; j < m; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                sum = sum + q_begin[k] * k_matrix[k * m + j];
            }
            s_begin[j] = sum;
            sum = 0;
        }
    }
    //softmax(q*k^t)
    float sum_softmax = 0.0f;
    float max_value = -INFINITY;
    //m行，一行一行处理
    for(int i = 0; i < m; ++i)
    {
        // max_value = std::max(max_value, );
        float *s_begin = s_matrix + i * m;
        for(int j = 0; j < m; ++j)
        {
            max_value = std::max(s_begin[j], max_value);
        }

        //sum
        for(int j = 0; j < m; ++j)
        {
            sum_softmax = sum_softmax + std::exp(s_begin[j] - max_value);
        }

        //softmax
        float inv_sum = 1.0f / sum_softmax;
        for(int j = 0; j < m; ++j)
        {
            s_begin[j] = std::exp(s_begin[j] - max_value) * inv_sum;
        }
        sum_softmax = 0.0f;
        max_value = -INFINITY;
    }
    //softmax(q*k^t) * v
    //size_s:[m, m], size_v:[m, n]
    for(int i = 0; i < m; ++i)
    {   
        float *s_begin = s_matrix + i * m;
        float *output_matrix_begin = output_matrix + i * n;
        float sum = 0.0f;
        for(int j = 0; j < n; ++j)
        {
            for(int k = 0; k < m; ++k)
            {
                sum = sum + s_begin[k] * v_matrix[j + k * n];
            }
            output_matrix_begin[j] = sum;
            sum = 0.0f;
        }
    }
    delete[] s_matrix;
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


int main()
{
    const int M = 1024;
    const int N = 512;
    const int element_size = M * N;

    //这里假设k_matrix已经是转置之后的
    float *Q_Matrix_Cpu = new float[element_size](); 
    float *K_Matrix_Cpu = new float[element_size]();
    float *V_Matrix_Cpu = new float[element_size]();
    float *output_cpu = new float[element_size]();

    for(int i = 0; i < M * N; i++)
    {
        Q_Matrix_Cpu[i] = rand() * 1.0f / RAND_MAX;
        K_Matrix_Cpu[i] = rand() * 1.0f / RAND_MAX;
        V_Matrix_Cpu[i] = rand() * 1.0f / RAND_MAX;
    }

    attention_cpu(Q_Matrix_Cpu, K_Matrix_Cpu, V_Matrix_Cpu, output_cpu, M, N);

    float *Q_Matrix_Gpu = new float[element_size](); 
    float *K_Matrix_Gpu = new float[element_size]();
    float *S_Matrix_Gpu = new float[M * M]();
    float *S_softmax= new float[M * M]();
    float *V_Matrix_Gpu = new float[element_size]();
    float *output_gpu = new float[element_size]();

    cudaMalloc(&Q_Matrix_Gpu, sizeof(float) * element_size);
    cudaMalloc(&K_Matrix_Gpu, sizeof(float) * element_size);
    cudaMalloc(&V_Matrix_Gpu, sizeof(float) * element_size);
    cudaMalloc(&output_gpu, sizeof(float) * element_size);
    cudaMalloc(&S_Matrix_Gpu, sizeof(float) * M * M);
    cudaMalloc(&S_softmax, sizeof(float) * M * M);

    cudaMemcpy(Q_Matrix_Gpu, Q_Matrix_Cpu, sizeof(float) * element_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K_Matrix_Gpu, K_Matrix_Cpu, sizeof(float) * element_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_Matrix_Gpu, V_Matrix_Cpu, sizeof(float) * element_size, cudaMemcpyHostToDevice);
    cudaMemset(output_gpu, 0, sizeof(float) * element_size);
    cudaMemset(S_Matrix_Gpu, 0, sizeof(float) * M * M);
    cudaMemset(S_softmax, 0, sizeof(float) * M * M);

    //q*k^t
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    sgemm<<<grid, block>>>(Q_Matrix_Gpu, K_Matrix_Gpu, S_Matrix_Gpu, M, N, M);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    //softmax(q * k^t)
    int block_soft = 16;
    int gird_soft = (M + block_soft - 1) / block_soft;
    softmax<<<gird_soft, block_soft>>>(S_Matrix_Gpu, S_softmax, M, M);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    //s_softmax * v
    dim3 block_atten(BLOCK, BLOCK);
    dim3 grid_atten((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    sgemm<<<grid_atten, block_atten>>>(S_softmax, V_Matrix_Gpu, output_gpu, M, M, N);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float *res = new float[element_size]();

    cudaMemcpy(res, output_gpu, sizeof(float) * element_size, cudaMemcpyDeviceToHost);

    view_result(output_cpu, res, M, N);

    if(check(res, output_cpu, M, N))
    {
        std::cout << "the result is right..." << std::endl;
    }
    else
    {
        std::cout << "the result is error..." << std::endl;
    }

    //delete pointer

    cudaFree(Q_Matrix_Gpu);
    cudaFree(K_Matrix_Gpu);
    cudaFree(V_Matrix_Gpu);
    cudaFree(output_gpu);
    cudaFree(S_Matrix_Gpu);
    cudaFree(S_softmax);

    delete []Q_Matrix_Cpu;
    delete []K_Matrix_Cpu;
    delete []V_Matrix_Cpu;
    delete []output_cpu;
    delete []res;
}