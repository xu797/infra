#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iomanip>   // std::setprecision
#include <cuda_runtime.h>

__global__ void flash_atten_v1(const float *Q, const float *K, const float *V, 
                                const int batch_size, const int num_head,
                                const int N, const int d,
                                const int Tc, const int Tr, const int Bc, const int Br, 
                                const float softmax_scale,
                                float *l, float *m, float *O)
{

    //Q_matrix： Tr 个分块,每个分块形状为 [Br, d]
    //kv_matrix:  Tc 个分块, 每个分块形状为 [Bc, d]
    //每个block处理一个N * d
    int tx = threadIdx.x;
    int bx = blockIdx.x; //batch_size
    int by = blockIdx.y; //num_head

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S = &sram[tile_size * 3]; // Bc * Br

    // block 每次处理的外循环是kv的块数，也就是Tc
    for(int i = 0; i < Tc; ++i)
    {
        //load kv to sram
        //每个block的thread_num == Bc, 所以刚好一个thread一行
        for(int j = 0; j < d; ++j)
        {
            Kj[tx * d + j] = K[qkv_offset + i * Bc * d + tx * d + j];
            Vj[tx * d + j] = V[qkv_offset + i * Bc * d + tx * d + j];
        }

        __syncthreads();
        //内层q的循环
        for(int j = 0; j < Tr; ++j)
        {
            // load q
            for(int p = 0; p < d; ++p)
            {
                Qi[tx * d + p] = Q[qkv_offset + j * Br * d + tx * d + p];
            }
            //拿到每个线程对应的行的l和m，l就是sum， m就是max_value
            float pre_l = l[lm_offset + j * Br + tx];
            float pre_m = m[lm_offset + j * Br + tx];

            //q*k^t 就是q的每一行和k的每一行相乘即可. 
            // 同样是一个线程负责q的一行和k的Bc行相乘
            float row_m = -INFINITY; //记录该行的最大值
            for(int p = 0; p < Bc; ++p)
            {
                float sum = 0.0f;                
                for(int q = 0; q < d; ++q)
                {
                    sum += Qi[tx * d + q] * Kj[p * d + q];
                }
                sum *= softmax_scale;
                S[tx * Bc + p] = sum; 
                row_m = fmaxf(sum, row_m);
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for(int p =0; p < Bc; ++p)
            {
                S[tx * Bc + p] = expf(S[tx * Bc + p] - row_m);
                row_l += S[tx * Bc + p];
            }

            //update l, m
            float row_m_new = fmaxf(row_m, pre_m);
            float row_l_new = expf(pre_m - row_m_new) * pre_l + expf(row_m - row_m_new) * row_l;

            //write o,l,m to HBM
            for(int p = 0; p < d; ++p)
            {
                float sum = 0.0f;
                for(int q = 0; q < Bc; ++q)
                {
                   sum += S[tx * Bc + q] * Vj[q * d + p];
                }
                // O[tx * d + p] = sum;
                O[qkv_offset + j * Br * d + tx * d + p] = 
                    (1 / row_l_new) * ((pre_l * expf(pre_m - row_m_new) * O[qkv_offset + (Br * d * j) + (tx * d) + p]) + (expf(row_m - row_m_new) * sum));
            }
        
            l[lm_offset + j * Br + tx] = row_l_new;
            m[lm_offset + j * Br + tx] = row_m_new;
        }
        __syncthreads();
    }
}


// CPU naive attention: Q,K,V [B,H,N,d], output O [B,H,N,d]
void naive_attention_cpu(const float* Q, const float* K, const float* V,
                        int batch_size, int num_head, int N, int d,
                        float softmax_scale, float* O_out)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int h = 0; h < num_head; h++)
        {
            // offset for this (batch, head)
            int base = b * num_head * N * d + h * N * d;
            for (int i = 0; i < N; i++) // query token i
            {
                // Step1: compute S[i,j] = scale * Q[i] @ K[j]^T
                std::vector<float> S(N, 0.0f);
                float row_max = -1e9f;
                for (int j = 0; j < N; j++) // key token j
                {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++)
                    {
                        float q_val = Q[base + i * d + k];
                        float k_val = K[base + j * d + k];
                        dot += q_val * k_val;
                    }
                    S[j] = dot * softmax_scale;
                    if (S[j] > row_max)
                        row_max = S[j];
                }

                // Step2: softmax: exp(S - row_max) / sum(exp(...))
                float sum_exp = 0.0f;
                std::vector<float> P(N, 0.0f);
                for (int j = 0; j < N; j++)
                {
                    P[j] = expf(S[j] - row_max);
                    sum_exp += P[j];
                }
                for (int j = 0; j < N; j++)
                {
                    P[j] /= sum_exp;
                }

                // Step3: O[i] = sum_j P[i,j] * V[j]
                for (int k = 0; k < d; k++)
                {
                    float ov = 0.0f;
                    for (int j = 0; j < N; j++)
                    {
                        float v_val = V[base + j * d + k];
                        ov += P[j] * v_val;
                    }
                    O_out[base + i * d + k] = ov;
                }
            }
        }
    }
}

// 计算最大绝对误差
float max_abs_error(const float* a, const float* b, int len)
{
    float max_err = 0.0f;
    for(int i=0;i<len;i++)
    {
        float err = fabs(a[i] - b[i]);
        if(err > max_err) max_err = err;
    }
    return max_err;
}

int main()
{
    // ========= 1. 超参配置 =========
    constexpr int batch_size = 1;
    constexpr int num_head = 1;
    constexpr int N = 512;    // seq_len，必须是 Br、Bc 的倍数
    constexpr int d = 1024;   // head dim
    constexpr int Bc = 2;
    constexpr int Br = 2;
    const int Tr = N / Br;
    const int Tc = N / Bc;
    const float softmax_scale = 1.0f / sqrtf((float)d);
    // 校验整除条件（kernel要求）
    if(N % Br != 0 || N % Bc !=0){
        std::cerr << "N must be divisible by Br and Bc!\n";
        return -1;
    }
    // ========= 2. 内存大小计算 =========
    size_t size_qkv = (size_t)batch_size * num_head * N * d * sizeof(float);
    size_t size_lm  = (size_t)batch_size * num_head * N * sizeof(float);
    size_t size_O   = size_qkv;
    float *h_Q = new float[batch_size * num_head * N * d]{};
    float *h_K = new float[batch_size * num_head * N * d]{};
    float *h_V = new float[batch_size * num_head * N * d]{};
    float *h_O = new float[batch_size * num_head * N * d]{};
    float *h_l = new float[batch_size * num_head * N]{};
    float *h_m = new float[batch_size * num_head * N]{};
    // 随机初始化 Q K V
    srand(42);
    for(int i=0; i < batch_size * num_head * N * d; i++){
        h_Q[i] = (float)(rand()) / RAND_MAX;
        h_K[i] = (float)(rand()) / RAND_MAX;
        h_V[i] = (float)(rand()) / RAND_MAX;
    }
    // l,m初始值：m=-inf，l=0（FlashAttention 在线softmax初始状态）
    for(int i=0; i < batch_size * num_head * N; i++){
        h_m[i] = -1e9f;
        h_l[i] = 0.0f;
    }
    // ========= 3. Device 显存分配 =========
    float *d_Q=nullptr, *d_K=nullptr, *d_V=nullptr;
    float *d_O=nullptr, *d_l=nullptr, *d_m=nullptr;
    cudaMalloc(&d_Q, size_qkv);
    cudaMalloc(&d_K, size_qkv);
    cudaMalloc(&d_V, size_qkv);
    cudaMalloc(&d_O, size_O);
    cudaMalloc(&d_l, size_lm);
    cudaMalloc(&d_m, size_lm);
    // Host -> Device
    cudaMemcpy(d_Q, h_Q, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, size_O, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, size_lm, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, size_lm, cudaMemcpyHostToDevice);
    // ========= 4. Shared memory 计算 & 打印信息 =========
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory per block: %d bytes\n", max_sram_size);
    printf("Requested shared memory: %d bytes\n", sram_size);
    if(sram_size > max_sram_size){
        std::cerr << "Requested shared memory exceeds device limit!\n";
        return -1;
    }
    // ========= 5. Launch kernel =========
    dim3 grid_dim(batch_size, num_head);
    std::cout << "Launch kernel: grid(" << grid_dim.x << "," << grid_dim.y << "), block(" << Bc << ")\n";
    dim3 block_dim(Bc);
    flash_atten_v1<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V,
        batch_size, num_head, N, d,
        Tc, Tr, Bc, Br, softmax_scale,
        d_l, d_m, d_O
    );
    // 检查kernel launch错误
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    // 等待GPU执行完成
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "Kernel finished.\n";
    // ========= 6. Device -> Host 取回结果 =========
    cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l, d_l, size_lm, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, size_lm, cudaMemcpyDeviceToHost);

    // ===================== 【新增：CPU Naive校验逻辑，只加在这里】 =====================
    std::vector<float> h_O_cpu(batch_size * num_head * N * d, 0.0f);
    naive_attention_cpu(h_Q, h_K, h_V, batch_size, num_head, N, d, softmax_scale, h_O_cpu.data());

    int total_elem = batch_size * num_head * N * d;
    float max_err = max_abs_error(h_O, h_O_cpu.data(), total_elem);
    std::cout << "\n===== Result Compare (GPU Flash vs CPU Naive) =====" << std::endl;
    std::cout << "Max absolute error: " << std::fixed << std::setprecision(8) << max_err << std::endl;

    // 打印前8个元素对比
    std::cout << "Index\tGPU_h_O\t\tCPU_h_O_cpu\n";
    for(int i=0;i<8;i++){
        std::cout << i << "\t" << h_O[i] << "\t" << h_O_cpu[i] << "\n";
    }
    // ==================================================================================

    // ========= 7. 资源释放 =========
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_O; delete[] h_l; delete[] h_m;
    cudaDeviceReset();
    return 0;
}
