#ifndef BFS_H
#define BFS_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GPU_TIMING_DEF
#define GPU_TIMING_DEF
typedef struct {
    float h2d_ms;      /* CSR build + host->device upload + dist init */
    float compute_ms;  /* the level exploration loop */
    float d2h_ms;      /* copying final dist[] back to host */
    float total_ms;    /* h2d_ms + compute_ms + d2h_ms */
    int   rounds;      /* number of levels explored */
} GpuTiming;
#endif

/* 1. Sequential CPU BFS (Queue-based) */
void bfs_cpu(const CSRGraph *graph, int source, int *distance);

/* 2. GPU Strategy 1: Thread-per-frontier-vertex */
int bfs_gpu_t_per_v(const CSRGraph *graph, int source, int *distance, GpuTiming *timing);

/*
 * =========================================================================
 * FUTURE BFS ALGORITHMS (To be added once implemented)
 * =========================================================================
 * TODO (Niharika): OpenMP CPU BFS
 * int bfs_openmp(const CSRGraph *graph, int source, int *distance);
 *
 * TODO (Niharika): GPU Strategy 2: Warp-per-frontier-vertex
 * int bfs_gpu_warp_per_v(const CSRGraph *graph, int source, int *distance, GpuTiming *timing);
 *
 * TODO (Niharika): GPU Strategy 3: Edge-based frontier mapping
 * int bfs_gpu_edge_based(const CSRGraph *graph, int source, int *distance, GpuTiming *timing);
 * =========================================================================
 */

/* Backward-compatible wrapper calling bfs_gpu_t_per_v */
void bfs_gpu(const CSRGraph *graph, int source, int *distance);

#ifdef __cplusplus
}
#endif

#endif
