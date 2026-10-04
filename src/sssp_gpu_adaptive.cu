//sssp_gpu_adaptive.cu

#include <cstdio>
#include <cstdlib>
#include "cuda_utils.h"
#include "csr.h"
#include "sssp.h"

#define DEGREE_THRESHOLD 32

/* LOW-DEGREE KERNEL: One thread per vertex.
   Only processes vertices with degree < DEGREE_THRESHOLD. */
__global__ void relax_kernel_adaptive_low(int* row_offset, int* col_idx, float* weights,
                                          float* dist, int V, int* changed_flag) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    int degree = row_offset[u + 1] - row_offset[u];
    if (degree >= DEGREE_THRESHOLD) return; // Delegate to high-degree kernel

    for (int k = row_offset[u]; k < row_offset[u + 1]; k++) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}

/* HIGH-DEGREE KERNEL: One warp (32 threads) per vertex.
   Only processes vertices with degree >= DEGREE_THRESHOLD. */
__global__ void relax_kernel_adaptive_high(int* row_offset, int* col_idx, float* weights,
                                           float* dist, int V, int* changed_flag) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    int u = global_warp_id;
    if (u >= V) return;

    int start_edge = row_offset[u];
    int end_edge = row_offset[u + 1];
    int degree = end_edge - start_edge;
    
    if (degree < DEGREE_THRESHOLD) return; // Delegate to low-degree kernel

    for (int k = start_edge + lane_id; k < end_edge; k += 32) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}

extern "C" int sssp_bellman_ford_gpu_adaptive(const CSRGraph *g, int source, float *dist, GpuTiming *timing) {
    int V = g->V;
    if (V <= 0) return 1;

    cudaEvent_t ev_start, ev_after_h2d, ev_after_compute, ev_after_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_after_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_after_compute));
    CUDA_CHECK(cudaEventCreate(&ev_after_d2h));

    CUDA_CHECK(cudaEventRecord(ev_start));

    DeviceCSR d_csr = copy_csr_to_device(g);

    float* d_dist;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));

    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    for (int i = 0; i < V; i++) dist[i] = SSSP_INF;
    dist[source] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_after_h2d));

    int blockSize = 256;
    int numBlocksLow = (V + blockSize - 1) / blockSize;
    int warps_needed = V;
    int threads_needed_high = warps_needed * 32;
    int numBlocksHigh = (threads_needed_high + blockSize - 1) / blockSize;

    int iter = 0;
    int h_changed = 1;
    while (iter < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch both kernels - they naturally partition the graph based on degree
        relax_kernel_adaptive_low<<<numBlocksLow, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                               d_csr.weights, d_dist, V, d_changed);
        relax_kernel_adaptive_high<<<numBlocksHigh, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                                d_csr.weights, d_dist, V, d_changed);
                                                                
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        iter++;
    }
    int rounds_to_converge = iter;

    h_changed = 0;
    CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
    relax_kernel_adaptive_low<<<numBlocksLow, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                           d_csr.weights, d_dist, V, d_changed);
    relax_kernel_adaptive_high<<<numBlocksHigh, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                            d_csr.weights, d_dist, V, d_changed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_compute));

    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_d2h));
    CUDA_CHECK(cudaEventSynchronize(ev_after_d2h));

    if (timing) {
        CUDA_CHECK(cudaEventElapsedTime(&timing->h2d_ms, ev_start, ev_after_h2d));
        CUDA_CHECK(cudaEventElapsedTime(&timing->compute_ms, ev_after_h2d, ev_after_compute));
        CUDA_CHECK(cudaEventElapsedTime(&timing->d2h_ms, ev_after_compute, ev_after_d2h));
        CUDA_CHECK(cudaEventElapsedTime(&timing->total_ms, ev_start, ev_after_d2h));
        timing->rounds = rounds_to_converge;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_h2d);
    cudaEventDestroy(ev_after_compute);
    cudaEventDestroy(ev_after_d2h);

    free_device_csr(&d_csr);
    cudaFree(d_dist);
    cudaFree(d_changed);

    if (h_changed) {
        fprintf(stderr, "Bellman-Ford (GPU Adaptive): negative-weight cycle detected.\n");
        return 0;
    }
    return 1;
}
