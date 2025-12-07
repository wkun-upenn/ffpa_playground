#include <cstdio>
#include <cuda_runtime.h>

__global__ void smem_bw_kernel(float *out, int iters, int stride) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = tid * stride;

    // Init shared mem
    smem[idx] = float(tid);
    __syncthreads();

    float val = smem[idx];

    // Each iteration: 1 load + 1 store to shared mem
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        val = val + smem[idx];
        smem[idx] = val;
    }

    // Prevent compiler from optimizing everything away
    out[blockIdx.x * blockDim.x + tid] = val;
}

int main() {
    int blocks = 256;          // tune based on SM count (H100 has many SMs)
    int threads = 256;         // threads per block
    int iters = 100000;        // big enough to dominate timing overhead
    int stride = 1;            // >1 if you want to avoid bank conflicts

    size_t shm_elems = threads * stride;
    size_t shm_bytes = shm_elems * sizeof(float);

    float *d_out;
    cudaMalloc(&d_out, blocks * threads * sizeof(float));

    // Warm-up launch
    smem_bw_kernel<<<blocks, threads, shm_bytes>>>(d_out, iters, stride);
    cudaDeviceSynchronize();

    // Time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    smem_bw_kernel<<<blocks, threads, shm_bytes>>>(d_out, iters, stride);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Each iteration: 1 load + 1 store per thread -> 2 accesses
    double accesses_per_iter = 2.0;
    double bytes_per_access = sizeof(float);

    double total_bytes =
        double(blocks) *
        double(threads) *
        double(iters) *
        accesses_per_iter *
        bytes_per_access;

    double seconds = ms / 1e3;
    double gb_per_s = (total_bytes / seconds) / 1e9;

    printf("SMEM kernel time: %.3f ms\n", ms);
    printf("Estimated SMEM throughput: %.2f GB/s\n", gb_per_s);

    cudaFree(d_out);
    return 0;
}