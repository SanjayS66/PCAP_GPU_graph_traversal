#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sssp.h"

/* Atomic min for a float: performs  if (value < *addr) *addr = value;  as one
   indivisible step, so two threads relaxing the same vertex cannot clobber each
   other's update. This is the CPU twin of the atomicMin used on the GPU side. */
static inline void atomic_min_float(float *addr, float value) {
    int32_t *iaddr = (int32_t *)addr;
    int32_t old = __atomic_load_n(iaddr, __ATOMIC_RELAXED);
    float old_f; memcpy(&old_f, &old, sizeof(float));
    while (value < old_f) {                       /* only try if we'd improve it */
        int32_t newbits; memcpy(&newbits, &value, sizeof(float));
        if (__atomic_compare_exchange_n(iaddr, &old, newbits, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;                               /* success: our value is stored */
        memcpy(&old_f, &old, sizeof(float));      /* lost the race; refresh + retry */
    }
}

int sssp_bellman_ford_openmp(const CSRGraph *g, int source, float *dist) {
    int V = g->V;

    #pragma omp parallel for
    for (int i = 0; i < V; i++)
        dist[i] = SSSP_INF;
    dist[source] = 0.0f;

    for (int iter = 0; iter < V - 1; iter++) {
        int changed = 0;

        /* Parallel vertex sweep. schedule(dynamic) balances the load because
           vertex degrees are very uneven (a few hub vertices, many small ones)
           -- the same load-imbalance issue the synopsis studies on the GPU.
           reduction(||:changed) safely ORs each thread's flag together. */
        #pragma omp parallel for schedule(dynamic, 256) reduction(||:changed)
        for (int u = 0; u < V; u++) {
            if (dist[u] == SSSP_INF) continue;
            for (int e = g->row_offset[u]; e < g->row_offset[u + 1]; e++) {
                int   v = g->col_index[e];
                float w = g->weights[e];
                if (dist[u] + w < dist[v]) {
                    atomic_min_float(&dist[v], dist[u] + w);
                    changed = 1;
                }
            }
        }

        if (!changed) break;
    }

    /* Negative-cycle check (single pass; cheap, keep it sequential). */
    for (int u = 0; u < V; u++) {
        if (dist[u] == SSSP_INF) continue;
        for (int e = g->row_offset[u]; e < g->row_offset[u + 1]; e++) {
            int v = g->col_index[e]; float w = g->weights[e];
            if (dist[u] + w < dist[v]) {
                fprintf(stderr, "Bellman-Ford (OpenMP): negative-weight cycle detected.\n");
                return 0;
            }
        }
    }
    return 1;
}