// bfs_gpu_edge_based.cu
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.h"
#include "csr.h"
#include "bfs.h"

/* One thread per EDGE. Edge k goes src_vertex[k] -> col_index[k]. If the
   source end is on the current level, try to claim the destination. */
__global__ void bfs_edge_kernel(int *src_vertex, int *col_index,
                                int *dist, int E, int level, int *changed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= E) return;

    int u = src_vertex[k];
    if (dist[u] != level) return;

    int v = col_index[k];
    if (atomicCAS(&dist[v], -1, level + 1) == -1)
        atomicExch(changed, 1);
}

extern "C" int bfs_gpu_edge_based(const CSRGraph *graph, int source,
                                  int *distance, GpuTiming *timing)
{
    int V = graph->V;
    int E = graph->E;
    if (V <= 0) return 1;

    cudaEvent_t ev_start, ev_h2d, ev_compute, ev_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_compute));
    CUDA_CHECK(cudaEventCreate(&ev_d2h));
    CUDA_CHECK(cudaEventRecord(ev_start));

    DeviceCSR_t_per_e d = copy_csr_to_device_t_per_e(graph);  /* builds src_vertex[] */

    int *d_dist, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    for (int i = 0; i < V; i++) distance[i] = -1;
    distance[source] = 0;
    CUDA_CHECK(cudaMemcpy(d_dist, distance, V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d));

    int blockSize = 256;
    int numBlocks = (E + blockSize - 1) / blockSize;

    int level = 0, h_changed = 1;
    while (h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        bfs_edge_kernel<<<numBlocks, blockSize>>>(d.src_vertex, d.col_index,
                                                  d_dist, E, level, d_changed);
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
    free_device_csr_t_per_e(&d);
    cudaFree(d_dist); cudaFree(d_changed);
    return 1;
}