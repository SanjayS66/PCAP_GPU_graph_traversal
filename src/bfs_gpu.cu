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
    int *frontier,
    int frontier_size,
    int *next_frontier,
    int *next_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= frontier_size)
        return;

    int vertex = frontier[idx];

    for (int i = row_offset[vertex]; i < row_offset[vertex + 1]; i++)
    {
        int neighbour = col_index[i];

        /* Visit only once */
        if (atomicCAS(&distance[neighbour], -1, distance[vertex] + 1) == -1)
        {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = neighbour;
        }
    }
}

/* ---------------- GPU BFS ---------------- */

extern "C" void bfs_gpu(const CSRGraph *graph, int source, int *distance)
{
    int V = graph->V;

    /* Device pointers */
    int *d_row_offset, *d_col_index;
    int *d_distance;
    int *d_frontier, *d_next_frontier;
    int *d_next_size;

    cudaMalloc(&d_row_offset, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_index, graph->E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_frontier, V * sizeof(int));
    cudaMalloc(&d_next_frontier, V * sizeof(int));
    cudaMalloc(&d_next_size, sizeof(int));

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

    cudaMemcpy(d_frontier, &source,
               sizeof(int), cudaMemcpyHostToDevice);

    int frontier_size = 1;

    while (frontier_size > 0)
    {
        cudaMemset(d_next_size, 0, sizeof(int));

        int threads = 256;
        int blocks = (frontier_size + threads - 1) / threads;

        bfs_kernel<<<blocks, threads>>>(
            V,
            d_row_offset,
            d_col_index,
            d_distance,
            d_frontier,
            frontier_size,
            d_next_frontier,
            d_next_size);

        cudaDeviceSynchronize();

        cudaMemcpy(&frontier_size, d_next_size,
                   sizeof(int), cudaMemcpyDeviceToHost);

        /* Swap frontiers */
        int *temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;
    }

    cudaMemcpy(distance, d_distance,
               V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offset);
    cudaFree(d_col_index);
    cudaFree(d_distance);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_size);
}
