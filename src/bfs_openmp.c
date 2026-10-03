#include <stdio.h>
#include <stdlib.h>
#include "csr.h"
#include "bfs.h"

/* OpenMP CPU BFS (topology-driven, level-synchronous).
   Same hop-distance result as the sequential queue BFS, but each level's
   vertices are processed in parallel across CPU cores. */
int bfs_openmp(const CSRGraph *graph, int source, int *distance)
{
    if (graph == NULL || distance == NULL) return 0;
    if (source < 0 || source >= graph->V) return 0;

    int V = graph->V;

    #pragma omp parallel for
    for (int i = 0; i < V; i++)
        distance[i] = -1;
    distance[source] = 0;

    int level = 0;
    int changed = 1;

    while (changed) {
        changed = 0;
        /* schedule(dynamic) balances uneven work (hub vertices have many
           neighbours). Two threads may write distance[v] at once, but both
           write the SAME value (level+1), so the race is harmless. */
        #pragma omp parallel for schedule(dynamic, 256) reduction(||:changed)
        for (int u = 0; u < V; u++) {
            if (distance[u] != level) continue;
            for (int e = graph->row_offset[u]; e < graph->row_offset[u + 1]; e++) {
                int v = graph->col_index[e];
                if (distance[v] == -1) {
                    distance[v] = level + 1;
                    changed = 1;
                }
            }
        }
        level++;
    }
    return 1;
}