#ifndef BFS_H
#define BFS_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

void bfs_cpu(const CSRGraph *graph, int source, int *distance);

void bfs_gpu(const CSRGraph *graph, int source, int *distance);

void bfs_gpu_t_per_e(const CSRGraph *graph,
                     int source,
                     int *distance);

#ifdef __cplusplus
}
#endif

#endif
