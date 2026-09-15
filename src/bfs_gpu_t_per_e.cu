#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "bfs.h"
#include "csr.h"


/*
 * One CUDA thread per CSR edge.
 *
 * Edge k corresponds to:
 *
 *     src_vertex[k] -> col_index[k]
 *
 * During BFS level 'level', only edges whose source vertex
 * has distance == level are active.
 */
__global__ void bfs_kernel_t_per_e(
    const int *src_vertex,
    const int *col_index,
    int *distance,
    int E,
    int level,
    int *next_size,
    int *next_frontier)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= E)
        return;

    int u = src_vertex[k];
    int v = col_index[k];

    /*
     * Only process edges leaving the current BFS frontier.
     */
    if (distance[u] != level)
        return;

    /*
     * Visit v exactly once.
     *
     * If v was unvisited (-1), assign it the next level.
     */
    if (atomicCAS(&distance[v], -1, level + 1) == -1)
    {
        int pos = atomicAdd(next_size, 1);

        next_frontier[pos] = v;
    }
}


/*
 * GPU BFS using one CUDA thread per edge.
 */
extern "C"
void bfs_gpu_t_per_e(const CSRGraph *graph,
                     int source,
                     int *distance)
{
    int V = graph->V;
    int E = graph->E;

    /*
     * Build source-vertex array for every CSR edge.
     *
     * For every edge index k:
     *
     *     src_vertex[k] = source vertex of that edge
     */
    int *src_vertex =
        (int *)malloc(sizeof(int) * E);

    if (src_vertex == NULL && E > 0)
    {
        fprintf(stderr,
                "Error: failed to allocate src_vertex\n");
        exit(1);
    }

    for (int u = 0; u < V; u++)
    {
        for (int k = graph->row_offset[u];
             k < graph->row_offset[u + 1];
             k++)
        {
            src_vertex[k] = u;
        }
    }


    /*
     * Device arrays
     */
    int *d_src_vertex;
    int *d_col_index;
    int *d_distance;

    int *d_next_size;
    int *d_next_frontier;


    cudaMalloc(&d_src_vertex,
               E * sizeof(int));

    cudaMalloc(&d_col_index,
               E * sizeof(int));

    cudaMalloc(&d_distance,
               V * sizeof(int));

    cudaMalloc(&d_next_size,
               sizeof(int));

    cudaMalloc(&d_next_frontier,
               V * sizeof(int));


    /*
     * Copy graph to GPU.
     */
    cudaMemcpy(d_src_vertex,
               src_vertex,
               E * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_index,
               graph->col_index,
               E * sizeof(int),
               cudaMemcpyHostToDevice);


    /*
     * Initialize distances.
     */
    for (int i = 0; i < V; i++)
        distance[i] = -1;

    distance[source] = 0;

    cudaMemcpy(d_distance,
               distance,
               V * sizeof(int),
               cudaMemcpyHostToDevice);


    /*
     * BFS proceeds level by level.
     */
    int level = 0;
    int frontier_size = 1;

    while (frontier_size > 0)
    {
        /*
         * Number of vertices discovered in next level.
         */
        cudaMemset(d_next_size,
                   0,
                   sizeof(int));


        int threads = 256;

        int blocks =
            (E + threads - 1) / threads;


        /*
         * Launch ONE THREAD PER EDGE.
         */
        bfs_kernel_t_per_e<<<blocks, threads>>>(
            d_src_vertex,
            d_col_index,
            d_distance,
            E,
            level,
            d_next_size,
            d_next_frontier);


        cudaError_t err =
            cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA error: %s\n",
                    cudaGetErrorString(err));

            exit(1);
        }


        /*
         * Retrieve number of newly discovered vertices.
         */
        cudaMemcpy(&frontier_size,
                   d_next_size,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);


        level++;
    }


    /*
     * Copy final BFS distances back.
     */
    cudaMemcpy(distance,
               d_distance,
               V * sizeof(int),
               cudaMemcpyDeviceToHost);


    /*
     * Cleanup.
     */
    cudaFree(d_src_vertex);
    cudaFree(d_col_index);
    cudaFree(d_distance);
    cudaFree(d_next_size);
    cudaFree(d_next_frontier);

    free(src_vertex);
}
