#ifndef BFS_H
#define BFS_H

#include "csr.h"

void bfs_cpu(const CSRGraph *graph, int source, int *distance);
void bfs_gpu(const CSRGraph *graph, int source, int *distance);

#endif