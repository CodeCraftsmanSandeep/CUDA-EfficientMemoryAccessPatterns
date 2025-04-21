#include <cuda.h>

constexpr unsigned int min_work_per_thread  = 16;

constexpr unsigned int block_size_256_power = 8;
constexpr unsigned int block_size_256       = (1 << block_size_256_power);

constexpr unsigned int block_size_512_power = 9;
constexpr unsigned int block_size_512       = (1 << block_size_512_power);

constexpr unsigned int num_blocks_256_power = 8;
constexpr unsigned int num_blocks_256       = (1 << num_blocks_256_power);

constexpr unsigned int warp_size_power      = 5;
constexpr unsigned int warp_size            = (1 << warp_size_power);

// Custom atomicAdd for datatype: (long long int)
__device__ long long int atomicAdd(long long int* address, long long int val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + (unsigned long long)val);
    } while (assumed != old);

    return old;
}

template <const unsigned int block_size_power>
__global__ void warpStrideKernel(const unsigned int N, const int* __restrict__ C, const int* __restrict__ B, long long int* __restrict__ intermediary_res)
{
    const unsigned int global_thread_id    =    ((blockIdx.x << block_size_power) + threadIdx.x);
    const unsigned int global_warp_id      =    (global_thread_id >> warp_size_power);

    const unsigned int num_chuncks         =    (N >> warp_size_power);
    const unsigned int num_warps           =    (gridDim.x << (block_size_power - warp_size_power));
    const unsigned int chuncks_per_warp    =    (num_chuncks / num_warps);
    const unsigned int work_per_warp       =    chuncks_per_warp << warp_size_power;

    unsigned int iter                      =    ((work_per_warp * global_warp_id) + (threadIdx.x & 31));
    const unsigned int end                 =    (iter + work_per_warp);
    constexpr unsigned int warp_stride     =    warp_size;
    
    long long int partial_sum = 0;
    while(__builtin_expect(iter < end, 1))
    {
        int diff = C[iter] - B[iter];
        partial_sum += diff * diff;
        iter += warp_stride;
    }

    // Remaining work-items
    iter = work_per_warp * num_warps + global_thread_id;
    if(iter < N)
    {
        int diff = C[iter] - B[iter];
        partial_sum += diff * diff;
    }

    // Tree reduction
    partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, 16);                                               
    partial_sum += __shfl_down_sync(0x0000FFFF, partial_sum,  8);                                               
    partial_sum += __shfl_down_sync(0x000000FF, partial_sum,  4);                                               
    partial_sum += __shfl_down_sync(0x0000000F, partial_sum,  2);                                               
    partial_sum += __shfl_down_sync(0x00000003, partial_sum,  1);                                               
                                                                                                                
    extern __shared__ long long int warp_partial_sum[];                                                         
    const uint8_t lane_id = (threadIdx.x & 31);                                                                
    const uint8_t warp_id = (threadIdx.x >> 5);                                                                
                                                                                                                
    if (lane_id == 0) warp_partial_sum[warp_id] = partial_sum;                                                  
    __syncthreads();                                                                                            
                                                                                                                
    if (warp_id == 0){                                                                                         
        partial_sum = warp_partial_sum[lane_id];                                                               
                                                                                                                
        if constexpr (block_size_power >= 10)  partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, 16);    
        if constexpr (block_size_power >=  9)  partial_sum += __shfl_down_sync(0x0000FFFF, partial_sum,  8);    
        if constexpr (block_size_power >=  8)  partial_sum += __shfl_down_sync(0x000000FF, partial_sum,  4);    
        if constexpr (block_size_power >=  7)  partial_sum += __shfl_down_sync(0x0000000F, partial_sum,  2);    
        if constexpr (block_size_power >=  6)  partial_sum += __shfl_down_sync(0x00000003, partial_sum,  1);    
                                                                                                                
        if (lane_id == 0) atomicAdd(intermediary_res, partial_sum);                                                    
    }
}

__global__ void sqrtKernel (float* __restrict__ L2_norm, const long long int* __restrict__ intermediary_res)
{
    *L2_norm = sqrt((double)*intermediary_res);
}

void compute(const unsigned int N, const int* __restrict__ d_C, const int* __restrict__ d_B, long long int* __restrict__ d_temp_storage, float* __restrict__ d_L2_norm)
{
    cudaMemset(d_temp_storage, 0LL, sizeof(long long int));
    cudaMemset(d_L2_norm, 0.0f, sizeof(float));

    // Kernel invocation
    if (N < block_size_256 * min_work_per_thread)
    {
        warpStrideKernel <block_size_256_power> 
                <<< 1, block_size_256, warp_size * sizeof(long long int) >>> (N, d_C, d_B, d_temp_storage);
    }
    else if (N < num_blocks_256 * block_size_256 * min_work_per_thread) 
    {
         warpStrideKernel <block_size_256_power> 
                <<< (N >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(long long int) >>> (N, d_C, d_B, d_temp_storage); 
    }
    else if (N < num_blocks_256 * block_size_512 * min_work_per_thread) 
    {
        warpStrideKernel <block_size_256_power> 
                <<< num_blocks_256, block_size_256, warp_size * sizeof(long long int) >>> (N, d_C, d_B, d_temp_storage); 
    }
    else
    {
        warpStrideKernel <block_size_512_power> 
                <<< num_blocks_256, block_size_512, warp_size * sizeof(long long int) >>> (N, d_C, d_B, d_temp_storage);
    }

    sqrtKernel <<< 1, 1 >>> (d_L2_norm, d_temp_storage);
    cudaDeviceSynchronize();

    return;
}
