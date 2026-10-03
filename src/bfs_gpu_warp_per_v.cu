// bfs_gpu_warp_per_v.cu
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.h"
#include "csr.h"
#include "bfs.h"

/* One WARP (32 threads) per vertex. Only vertices at the current `level`
   expand; their 32 lanes split the neighbour list, so a high-degree vertex
   gets 32 workers instead of 1. */
__global__ void bfs_warp_kernel(int *row_offset, int *col_index,
                                int *dist, int V, int level, int *changed)
{
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    if (warp_id >= V) return;

    int u = warp_id;
    if (dist[u] != level) return;              /* only expand current level */

    for (int k = row_offset[u] + lane; k < row_offset[u + 1]; k += 32) {
        int v = col_index[k];
        /* claim v only if unvisited (-1); CAS makes it race-free */
        if (atomicCAS(&dist[v], -1, level + 1) == -1)
            atomicExch(changed, 1);
    }
}

extern "C" int bfs_gpu_warp_per_v(const CSRGraph *graph, int source,
                                  int *distance, GpuTiming *timing)
{
    int V = graph->V;
    if (V <= 0) return 1;

    cudaEvent_t ev_start, ev_h2d, ev_compute, ev_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_compute));
    CUDA_CHECK(cudaEventCreate(&ev_d2h));
    CUDA_CHECK(cudaEventRecord(ev_start));

    DeviceCSR d = copy_csr_to_device(graph);   /* uploads CSR arrays */

    int *d_dist, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    for (int i = 0; i < V; i++) distance[i] = -1;
    distance[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, distance, V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d));

    int blockSize = 256;
    long total_threads = (long)V * 32;
    int numBlocks = (int)((total_threads + blockSize - 1) / blockSize);

    int level = 0, h_changed = 1;
    while (h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        bfs_warp_kernel<<<numBlocks, blockSize>>>(d.row_offset, d.col_index,
                                                  d_dist, V, level, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    }

    CUDA_CHECK(cudaEventRecord(ev_compute));
    CUDA_CHECK(cudaMemcpy(distance, d_dist, V * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h));

    if (timing) {
        CUDA_CHECK(cudaEventElapsedTime(&timing->h2d_ms,     ev_start,   ev_h2d));
        CUDA_CHECK(cudaEventElapsedTime(&timing->compute_ms, ev_h2d,     ev_compute));
        CUDA_CHECK(cudaEventElapsedTime(&timing->d2h_ms,     ev_compute, ev_d2h));
        CUDA_CHECK(cudaEventElapsedTime(&timing->total_ms,   ev_start,   ev_d2h));
        timing->rounds = level;
    }

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_h2d);
    cudaEventDestroy(ev_compute); cudaEventDestroy(ev_d2h);
    free_device_csr(&d);
    cudaFree(d_dist); cudaFree(d_changed);
    return 1;
}