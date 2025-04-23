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

// Reduction kernel
template <const unsigned int block_size_power>
__global__ void gridStrideKernel (const unsigned int matrix_size, const int* __restrict__ C, long long int* __restrict__ intermediary_res)
{
    unsigned int iter = threadIdx.x + (blockIdx.x << block_size_power);
    const unsigned int grid_stride = gridDim.x << block_size_power;

    // Grid-stride access
    long long int partial_sum = 0;
    while(__builtin_expect(iter < matrix_size, 1)){    // Branch prediction: Always take
        partial_sum += C[iter] * C[iter];
        iter += grid_stride;
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

__global__ void sqrtKernel (float* __restrict__ Frobenius_norm, long long int* __restrict__ intermediary_res)
{
    *Frobenius_norm = sqrt((double)*intermediary_res);
}

void compute(const unsigned int M, const unsigned int N, const int* __restrict__ d_C, long long int* __restrict__ d_temp_storage, float* __restrict__ d_Frobenius_norm)
{
    cudaMemset(d_temp_storage, 0LL, sizeof(long long int));
    cudaMemset(d_Frobenius_norm, 0.0f, sizeof(float));
    const unsigned int matrix_size = M * N;

    // Kernel invocation
    if (matrix_size < block_size_256 * min_work_per_thread)
    {
        gridStrideKernel       <block_size_256_power> 
                <<< 1, block_size_256, warp_size * sizeof(long long int) >>> (matrix_size, d_C, d_temp_storage);
    }
    else if (matrix_size < num_blocks_256 * block_size_256 * min_work_per_thread) 
    {
        gridStrideKernel       <block_size_256_power>
                <<< (matrix_size >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(long long int) >>> (matrix_size, d_C, d_temp_storage);
    }
    else if (matrix_size < num_blocks_256 * block_size_512 * min_work_per_thread) 
    {
        gridStrideKernel       <block_size_256_power> 
               <<< num_blocks_256, block_size_256, warp_size * sizeof(long long int) >>> (matrix_size, d_C, d_temp_storage);
    }
    else
    {
        gridStrideKernel       <block_size_512_power> 
              <<< num_blocks_256, block_size_512, warp_size * sizeof(long long int) >>> (matrix_size, d_C, d_temp_storage);
    }
    sqrtKernel <<< 1, 1>>> (d_Frobenius_norm, d_temp_storage);
    cudaDeviceSynchronize();
    return;
}
