#include<iostream>
#include<cmath>
#include <iomanip>
#include<cuda_runtime.h>

// #define MAX_ELEMS_PER_THREAD 32

__device__ void warpReduceOnline(float &max_value, float &sum)
{   
    //offset 取值 16 8 4 2 1
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        //warp内部 0- 15 thread 拿到16-31 thread 的local_max, local_sum
        //下一次就是0-7 拿到 8-15的
        float other_max_value = __shfl_down_sync(0xffffffff, max_value, offset);
        float other_sum = __shfl_down_sync(0xffffffff, sum, offset);

        // 空分片直接跳过合并，防止 -inf - (-inf) → NaN
        if (other_max_value == -INFINITY && other_sum == 0.0f)
        {
            continue;
        }
            
        //归约合并
        //thread 0的local_max = max(thread 0的local_max, thread 16的local_max)
        float pre_max = max_value;
        max_value = fmaxf(max_value, other_max_value);
        sum = sum * expf(pre_max - max_value) + other_sum * expf(other_max_value - max_value);
    }
}

__global__ void softmax(float *input, float *output, int M, int N)
{   

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    //不再是一行一个block，现在一个block处理多行
    for(int i = blockIdx.x; i < M; i += gridDim.x)
    {
        float *input_begin = input + N * i;
        float *output_begin = output + N * i;

        float4 *input_begin_f4 = reinterpret_cast<float4*>(input_begin);
        float4 *output_begin_f4 = reinterpret_cast<float4*>(output_begin);

        float local_max;
        float local_pre_max = -INFINITY;
        float local_sum = 0.0f;
        // float reg_cache[MAX_ELEMS_PER_THREAD];
        // int count = 0;
        
        int N4 = N / 4; 

        for(int j = tid; j < N4; j += blockDim.x)
        {
            float vals[4] = {input_begin_f4[j].x, input_begin_f4[j].y,
                            input_begin_f4[j].z, input_begin_f4[j].w};

            for(int k = 0; k < 4; ++k)
            {
                local_max = fmaxf(local_pre_max, vals[k]);
                local_sum = local_sum * expf(local_pre_max - local_max) + expf(vals[k] - local_max);
                local_pre_max = local_max;
            }
        }

        for(int j = tid + N4 * 4; j < N; j += blockDim.x)
        {
            local_max = fmaxf(local_pre_max, input_begin[j]);
            local_sum = local_sum * expf(local_pre_max - local_max) + expf(input_begin[j] - local_max);
            local_pre_max = local_max;
        }

        // warp内部归约
        warpReduceOnline(local_max, local_sum);

        __shared__ float warp_max[32];
        __shared__ float warp_sum[32];

        if(lane_id == 0)
        {
            warp_max[warp_id] = local_max;
            warp_sum[warp_id] = local_sum; 
        }
        
        __syncthreads();

        int num_warps = N / blockDim.x;
        if(warp_id == 0)
        {
            local_max = (lane_id < num_warps) ? warp_max[lane_id] : -INFINITY;
            local_sum = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
            warpReduceOnline(local_max, local_sum);
        }

        __shared__ float final_max;
        __shared__ float final_sum;

        if(tid == 0)
        {
            final_max = local_max;
            final_sum = local_sum;
        }

        __syncthreads();
        
        float max_value = final_max;
        float inv_sum = 1.0f / final_sum;

        for(int j = tid; j < N4; j += blockDim.x)
        {
            float4 data = input_begin_f4[j];
            float4 result;
            result.x = expf(data.x - max_value) * inv_sum;
            result.y = expf(data.y - max_value) * inv_sum;
            result.z = expf(data.z - max_value) * inv_sum;
            result.w = expf(data.w - max_value) * inv_sum;

            output_begin_f4[j] = result;
        }

        for(int j = tid + N4 * 4; j < N; j += blockDim.x)
        {
            output_begin[j] = expf(input_begin[j] - max_value) * inv_sum;
        }

        __syncthreads();  // 确保 Shared Memory 在下一行之前被重置

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
            output[i * N + j] = std::exp(input[i * N + j] - max_value)/ sum;
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

int main()
{
    const int M = 512;
    const int N = 1024;
    float *input_cpu = new float[M * N]();
    float *output_cpu = new float[M * N]();

    for(int i = 0; i < M * N; i++)
    {
        input_cpu[i] = rand() * 1.0f / RAND_MAX;
    }

    softmax_cpu(input_cpu, output_cpu, M, N);

    float *input_gpu;
    float *output_gpu;

    cudaMalloc(&input_gpu, sizeof(float) * M * N);
    cudaMalloc(&output_gpu, sizeof(float) * M * N);

    cudaMemcpy(input_gpu, input_cpu, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemset(output_gpu, 0, sizeof(float) * M * N);

    int block_size = 128;
    // int grid_size = (M + block_size - 1) / block_size;
    int grid_size = M / 3;
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

    view_result(output_cpu, res, M, N);

    if(check(res, output_cpu, M, N))
    {
        std::cout << "the result is right..." << std::endl;
    }
    else
    {
        std::cout << "the result is error..." << std::endl;
    }


    delete []input_cpu;
    delete []output_cpu;
    delete []res;

    cudaFree(input_gpu);
    cudaFree(output_gpu);
}