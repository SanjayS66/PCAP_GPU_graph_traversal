#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "bfs.h"
/* ---------------- GPU Kernel ---------------- */

__global__ void bfs_kernel(
    int V,
    int *row_offset,
    int *col_index,
    int *distance,
    int level,
    int *changed)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    if (distance[u] != level) return;

    for (int i = row_offset[u]; i < row_offset[u + 1]; i++)
    {
        int v = col_index[i];

        /* Non-atomic benign race condition write */
        if (distance[v] == -1)
        {
            distance[v] = level + 1;
            *changed = 1;
        }
    }
}

/* ---------------- GPU BFS (Thread-per-frontier-vertex) ---------------- */

extern "C" int bfs_gpu_t_per_v(const CSRGraph *graph, int source, int *distance, GpuTiming *timing)
{
    int V = graph->V;
    if (V <= 0) return 1;

    cudaEvent_t ev_start, ev_after_h2d, ev_after_compute, ev_after_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_after_h2d);
    cudaEventCreate(&ev_after_compute);
    cudaEventCreate(&ev_after_d2h);

    cudaEventRecord(ev_start);

    /* Device pointers */
    int *d_row_offset, *d_col_index;
    int *d_distance;
    int *d_changed;
    cudaMalloc(&d_row_offset, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_index, graph->E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_row_offset, graph->row_offset,
               (V + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_index, graph->col_index,
               graph->E * sizeof(int), cudaMemcpyHostToDevice);

    /* Initialize distances */
    for (int i = 0; i < V; i++)
        distance[i] = -1;

    distance[source] = 0;

    cudaMemcpy(d_distance, distance,
               V * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(ev_after_h2d);

    int level = 0;
    int h_changed = 1;

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    while (h_changed)
    {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        bfs_kernel<<<blocks, threads>>>(
            V,
            d_row_offset,
            d_col_index,
            d_distance,
            level,
            d_changed);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed,
                   sizeof(int), cudaMemcpyDeviceToHost);

        level++;
    }

    cudaEventRecord(ev_after_compute);

    cudaMemcpy(distance, d_distance,
               V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(ev_after_d2h);
    cudaEventSynchronize(ev_after_d2h);

    if (timing) {
        cudaEventElapsedTime(&timing->h2d_ms,     ev_start,         ev_after_h2d);
        cudaEventElapsedTime(&timing->compute_ms, ev_after_h2d,     ev_after_compute);
        cudaEventElapsedTime(&timing->d2h_ms,     ev_after_compute, ev_after_d2h);
        cudaEventElapsedTime(&timing->total_ms,   ev_start,         ev_after_d2h);
        timing->rounds = level;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_h2d);
    cudaEventDestroy(ev_after_compute);
    cudaEventDestroy(ev_after_d2h);

    cudaFree(d_row_offset);
    cudaFree(d_col_index);
    cudaFree(d_distance);
    cudaFree(d_changed);

    return 1;
}

/* Backward-compatible wrapper */
extern "C" void bfs_gpu(const CSRGraph *graph, int source, int *distance)
{
    bfs_gpu_t_per_v(graph, source, distance, NULL);
}
