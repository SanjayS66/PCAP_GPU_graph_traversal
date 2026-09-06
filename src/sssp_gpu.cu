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

extern "C" int sssp_bellman_ford_gpu(const CSRGraph *g, int source, float *dist) {
    int V = g->V;
    if (V <= 0) return 1;

    /* copy_csr_to_device currently expects a non-const CSRGraph*; it only
       reads from it, so the cast is safe. Consider updating its signature
       to take `const CSRGraph*` for full const-correctness. */
    DeviceCSR d_csr = copy_csr_to_device((CSRGraph*)g);

    float* d_dist;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));

    int* d_changed;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    /* 1. Initialise into the caller-provided dist buffer, then push to
          device. Same init step as sssp_bellman_ford_cpu. */
    for (int i = 0; i < V; i++) dist[i] = SSSP_INF;
    dist[source] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;

    /* 2. Relax all edges up to V-1 times, same convergence rule as CPU. */
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

    /* 3. One extra pass: if anything can still relax, a negative-weight
          cycle exists. Mirrors the CPU version's final check pass. */
    h_changed = 0;
    CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
    relax_kernel<<<numBlocks, blockSize>>>(d_csr.row_offset, d_csr.col_index,
                                            d_csr.weights, d_dist, V, d_changed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    free_device_csr(&d_csr);
    cudaFree(d_dist);
    cudaFree(d_changed);

    if (h_changed) {
        fprintf(stderr, "Bellman-Ford (GPU): negative-weight cycle detected.\n");
        return 0;
    }
    return 1;
}