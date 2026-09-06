//sssp_gpu_t_per_v.cu

#include <cstdio>
#include <cstdlib>
#include "cuda_utils.h"
#include "csr.h"
#include "sssp.h"

/* One thread per vertex per round: scan u's outgoing edges and try to
   relax each neighbour v. Identical relaxation rule to the CPU version,
   just parallelised across vertices instead of looping sequentially. */
__global__ void relax_kernel(int* row_offset, int* col_idx, float* weights,
                              float* dist, int V, int* changed_flag) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    for (int k = row_offset[u]; k < row_offset[u + 1]; k++) {
        int v = col_idx[k];
        float w = weights[k];
        if (dist[u] + w < dist[v]) {
            atomicMinFloat(&dist[v], dist[u] + w);
            atomicExch(changed_flag, 1);
        }
    }
}

extern "C" int sssp_bellman_ford_gpu(const CSRGraph *g, int source, float *dist, GpuTiming *timing) {
    int V = g->V;
    if (V <= 0) return 1;

    /* --- Events for split timing: H2D transfer / compute / D2H transfer --- */
    cudaEvent_t ev_start, ev_after_h2d, ev_after_compute, ev_after_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_after_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_after_compute));
    CUDA_CHECK(cudaEventCreate(&ev_after_d2h));

    CUDA_CHECK(cudaEventRecord(ev_start));

    /* --- H2D: upload CSR arrays --- */
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
    int numBlocks = (V + blockSize - 1) / blockSize;

    /* --- Compute: relax all edges up to V-1 times --- */
    int iter = 0;
    int h_changed = 1;
    while (iter < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        relax_kernel<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                                d_csr.weights, d_dist, V, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        iter++;
    }
    int rounds_to_converge = iter;

    /* Extra pass: negative-weight cycle check. */
    h_changed = 0;
    CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
    relax_kernel<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                            d_csr.weights, d_dist, V, d_changed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_compute));

    /* --- D2H: copy final distances back --- */
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_d2h));
    CUDA_CHECK(cudaEventSynchronize(ev_after_d2h));

    /* --- Fill caller's timing breakdown, if requested --- */
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
        fprintf(stderr, "Bellman-Ford (GPU, thread-per-vertex): negative-weight cycle detected.\n");
        return 0;
    }
    return 1;
}
