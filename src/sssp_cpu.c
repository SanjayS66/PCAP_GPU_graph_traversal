#include <stdio.h>
#include "sssp.h"

int sssp_bellman_ford_cpu(const CSRGraph *g, int source, float *dist) {
    int V = g->V;

    /* 1. Initialise: every vertex is "infinitely" far, source is 0. */
    for (int i = 0; i < V; i++)
        dist[i] = SSSP_INF;
    dist[source] = 0.0f;

    /* 2. Relax all edges up to V-1 times.
          Relaxing edge u->v (weight w) means: if reaching v THROUGH u is
          cheaper than the best we know for v, update v's distance. */
    for (int iter = 0; iter < V - 1; iter++) {
        int changed = 0;                       /* did this pass improve anything? */

        for (int u = 0; u < V; u++) {
            if (dist[u] == SSSP_INF) continue; /* can't relax from an unreached vertex */

            /* u's neighbours live in col_index[start..end) in the CSR arrays. */
            int start = g->row_offset[u];
            int end   = g->row_offset[u + 1];
            for (int e = start; e < end; e++) {
                int   v = g->col_index[e];
                float w = g->weights[e];
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    changed = 1;
                }
            }
        }

        if (!changed) break;                   /* nothing improved => already optimal */
    }

    /* 3. One extra pass: if any edge can STILL relax, a negative cycle exists. */
    for (int u = 0; u < V; u++) {
        if (dist[u] == SSSP_INF) continue;
        for (int e = g->row_offset[u]; e < g->row_offset[u + 1]; e++) {
            int   v = g->col_index[e];
            float w = g->weights[e];
            if (dist[u] + w < dist[v]) {
                fprintf(stderr, "Bellman-Ford: negative-weight cycle detected.\n");
                return 0;
            }
        }
    }

    return 1;
}