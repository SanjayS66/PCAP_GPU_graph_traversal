#ifndef SSSP_H
#define SSSP_H

#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel meaning "unreachable / no path yet". A big float is simpler
   to compare and print than math.h's INFINITY. */
#define SSSP_INF 1e30f

/* Breakdown of where a GPU SSSP call spent its time. Filled in by the GPU
   variants below so the caller can report transfer-vs-compute separately
   instead of only a single opaque total. */
typedef struct {
    float h2d_ms;      /* CSR build + host->device upload + dist init */
    float compute_ms;  /* the relaxation round loop + final cycle-check pass */
    float d2h_ms;       /* copying final dist[] back to host */
    float total_ms;    /* h2d_ms + compute_ms + d2h_ms */
    int   rounds;       /* number of relaxation rounds actually run before convergence */
} GpuTiming;

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
     g       : graph in CSR form (host-side; device copies are made internally)
     source  : starting vertex (0-based)
     dist    : caller-allocated array of size g->V, filled with the shortest
               distance from source to each vertex (SSSP_INF if unreachable)
     timing  : optional (may be NULL) - if non-NULL, filled with the
               h2d/compute/d2h/total/rounds breakdown for this call
   Returns 1 on success, 0 if a negative-weight cycle is detected. */
int sssp_bellman_ford_gpu(const CSRGraph *g, int source, float *dist, GpuTiming *timing);

/* Same as above, one CUDA thread per edge instead of per vertex. */
int sssp_bellman_ford_gpu_t_per_e(const CSRGraph *g, int source, float *dist, GpuTiming *timing);

#ifdef __cplusplus
}
#endif

#endif /* SSSP_H */
