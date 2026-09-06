#ifndef SSSP_H
#define SSSP_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel meaning "unreachable / no path yet". A big float is simpler
   to compare and print than math.h's INFINITY. */
#define SSSP_INF 1e30f

/* Single-Source Shortest Path via Bellman-Ford (CPU reference version).
     g      : graph in CSR form
     source : starting vertex (0-based)
     dist   : caller-allocated array of size g->V, filled with the shortest
              distance from source to each vertex (SSSP_INF if unreachable)
   Returns 1 on success, 0 if a negative-weight cycle is detected. */
int sssp_bellman_ford_cpu(const CSRGraph *g, int source, float *dist);
/* Same algorithm, vertex sweep parallelised across CPU cores with OpenMP.
   Produces results identical to the sequential version. */
int sssp_bellman_ford_openmp(const CSRGraph *g, int source, float *dist);

/* Same algorithm again, one CUDA thread per vertex per relaxation round.
     g      : graph in CSR form (host-side; device copies are made internally)
     source : starting vertex (0-based)
     dist   : caller-allocated array of size g->V, filled with the shortest
              distance from source to each vertex (SSSP_INF if unreachable)
   Returns 1 on success, 0 if a negative-weight cycle is detected. */
int sssp_bellman_ford_gpu(const CSRGraph *g, int source, float *dist);
 
#ifdef __cplusplus
}
#endif

#endif /* SSSP_H */