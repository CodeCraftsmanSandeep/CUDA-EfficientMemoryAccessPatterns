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

// Reduction kernel
template <const unsigned int block_size_power>
__global__ void blockStrideKernel (const unsigned int N, const int* __restrict__ C, const int* __restrict__ B, int* __restrict__ Linfinity_norm)
{
    const unsigned int global_thread_id     =   (threadIdx.x + (blockIdx.x << block_size_power));
    const unsigned int num_chuncks          =   (N >> block_size_power);
    const unsigned int chuncks_per_block    =   (num_chuncks / gridDim.x);

    const unsigned int work_per_block       =   (chuncks_per_block << block_size_power);
    unsigned int iter                       =   ((work_per_block * blockIdx.x) + threadIdx.x);
    const unsigned int end                  =   (iter + work_per_block);
    constexpr unsigned int block_stride     =   (1 << block_size_power);

    // Block-stride access
    int partial_sum = 0;
    while(__builtin_expect(iter < end, 1)){
        partial_sum = max(partial_sum, abs(C[iter] - B[iter]));
        iter += block_stride;        
    }

    // Remaining work-items
    iter = ((work_per_block * gridDim.x) + global_thread_id);
    if(iter < N) partial_sum = max(partial_sum, abs(C[iter] - B[iter]));

    // Tree reduction
    partial_sum = max(partial_sum, __shfl_down_sync(0xFFFFFFFF, partial_sum, 16));
    partial_sum = max(partial_sum, __shfl_down_sync(0x0000FFFF, partial_sum,  8));
    partial_sum = max(partial_sum, __shfl_down_sync(0x000000FF, partial_sum,  4));
    partial_sum = max(partial_sum, __shfl_down_sync(0x0000000F, partial_sum,  2));
    partial_sum = max(partial_sum, __shfl_down_sync(0x00000003, partial_sum,  1));

    extern __shared__ int warp_partial_sum[];
    const uint8_t lane_id = (threadIdx.x & 31);
    const uint8_t warp_id = (threadIdx.x >> 5);

    if (lane_id == 0) warp_partial_sum[warp_id] = partial_sum;
    __syncthreads();

    if (warp_id == 0){
        partial_sum = warp_partial_sum[lane_id];

        if constexpr (block_size_power >= 10)  partial_sum = max(partial_sum, __shfl_down_sync(0xFFFFFFFF, partial_sum, 16));
        if constexpr (block_size_power >=  9)  partial_sum = max(partial_sum, __shfl_down_sync(0x0000FFFF, partial_sum,  8));
        if constexpr (block_size_power >=  8)  partial_sum = max(partial_sum, __shfl_down_sync(0x000000FF, partial_sum,  4));
        if constexpr (block_size_power >=  7)  partial_sum = max(partial_sum, __shfl_down_sync(0x0000000F, partial_sum,  2));
        if constexpr (block_size_power >=  6)  partial_sum = max(partial_sum, __shfl_down_sync(0x00000003, partial_sum,  1));

        if (lane_id == 0) atomicMax(Linfinity_norm, partial_sum);
    }
}

void compute(const unsigned int N, const int* __restrict__ d_C, const int* __restrict__ d_B, int* __restrict__ d_Linfinity_norm)
{
    cudaMemset(d_Linfinity_norm, 0, sizeof(int));

    // Kernel invocation
    if (N < block_size_256 * min_work_per_thread)
    {
        blockStrideKernel       <block_size_256_power> 
                <<< 1, block_size_256, warp_size * sizeof(int) >>> (N, d_C, d_B, d_Linfinity_norm);
    }
    else if (N < num_blocks_256 * block_size_256 * min_work_per_thread) 
    {
        blockStrideKernel       <block_size_256_power>
                <<< (N >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(int) >>> (N, d_C, d_B, d_Linfinity_norm);
    }
    else if (N < num_blocks_256 * block_size_512 * min_work_per_thread) 
    {
        blockStrideKernel       <block_size_256_power> 
               <<< num_blocks_256, block_size_256, warp_size * sizeof(int) >>> (N, d_C, d_B, d_Linfinity_norm);
    }
    else
    {
        blockStrideKernel       <block_size_512_power> 
              <<< num_blocks_256, block_size_512, warp_size * sizeof(int) >>> (N, d_C, d_B, d_Linfinity_norm);
    }

    cudaDeviceSynchronize();
}
