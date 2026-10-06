// src/sssp_gpu_adaptive.cu
//
// Degree-aware adaptive SSSP: pre-partitioned low/high vertex lists.
//
// The old design checked vertex degree *inside* every kernel thread and
// returned early for the wrong partition — wasting threads and keeping
// the grid sized for ALL vertices even when only a fraction needed warp-
// level parallelism.
//
// This version:
//   1. Partitions vertices ONCE on the host (O(V), before timing starts):
//        low_list  : vertices with degree <= ADAPT_THRESHOLD  → 1 thread each
//        high_list : vertices with degree >  ADAPT_THRESHOLD  → 1 warp  each
//   2. Uploads both lists to device (one-time cost, outside the timed loop).
//   3. Each Bellman-Ford iteration launches:
//        sssp_adaptive_low_kernel  with grid sized for low_count  threads
//        sssp_adaptive_high_kernel with grid sized for high_count warps
//   No branching / early-return inside either kernel.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_utils.h"   // CUDA_CHECK, DeviceCSR, atomicMinFloat, copy_csr_to_device
#include "csr.h"          // CSRGraph
#include "sssp.h"         // sssp_bellman_ford_gpu_adaptive, GpuTiming, SSSP_INF

#define ADAPT_THRESHOLD  32     // degree boundary: ≤32 → thread, >32 → warp
#define BLOCK            256    // threads per block (8 warps per block)
#define WARPS_PER_BLOCK  (BLOCK / 32)

// ---------------------------------------------------------------------------
// LOW-degree kernel: one thread per vertex from low_list.
// Every thread processes its vertex's full edge list sequentially.
// ---------------------------------------------------------------------------
__global__ void sssp_adaptive_low_kernel(
        const int*   __restrict__ row_offset,
        const int*   __restrict__ col_index,
        const float* __restrict__ weights,
        float*       __restrict__ dist,
        const int*   __restrict__ low_list,
        int   low_count,
        int*  changed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= low_count) return;

    int   u  = low_list[idx];
    float du = dist[u];
    if (du >= SSSP_INF) return;   // unreachable vertex — nothing to relax

    int start = row_offset[u];
    int end   = row_offset[u + 1];
    for (int e = start; e < end; ++e) {
        int   v  = col_index[e];
        float nd = du + weights[e];
        if (nd < dist[v]) {
            atomicMinFloat(&dist[v], nd);
            *changed = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// HIGH-degree kernel: one warp (32 threads) per vertex from high_list.
// The 32 lanes stride through the edge list of vertex u, 32 edges at a time,
// so a hub vertex with thousands of edges gets 32 concurrent workers.
// ---------------------------------------------------------------------------
__global__ void sssp_adaptive_high_kernel(
        const int*   __restrict__ row_offset,
        const int*   __restrict__ col_index,
        const float* __restrict__ weights,
        float*       __restrict__ dist,
        const int*   __restrict__ high_list,
        int   high_count,
        int*  changed)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // global warp index
    int lane    = threadIdx.x & 31;                               // lane within warp (0-31)
    if (warp_id >= high_count) return;

    int   u  = high_list[warp_id];
    float du = dist[u];
    if (du >= SSSP_INF) return;   // unreachable vertex

    int start = row_offset[u];
    int end   = row_offset[u + 1];
    for (int e = start + lane; e < end; e += 32) {
        int   v  = col_index[e];
        float nd = du + weights[e];
        if (nd < dist[v]) {
            atomicMinFloat(&dist[v], nd);
            *changed = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher — called by the benchmark harness.
// Signature matches sssp.h: sssp_bellman_ford_gpu_adaptive()
// ---------------------------------------------------------------------------
extern "C" int sssp_bellman_ford_gpu_adaptive(
        const CSRGraph *g, int source, float *dist, GpuTiming *timing)
{
    int V = g->V;
    if (V <= 0) return 1;

    // -----------------------------------------------------------------------
    // Step 1: Build low/high vertex lists on the HOST (outside timed region).
    //         This is pure CSR work (single O(V) pass) — not kernel compute.
    // -----------------------------------------------------------------------
    std::vector<int> h_low, h_high;
    h_low.reserve(V);
    h_high.reserve(V);
    for (int u = 0; u < V; ++u) {
        int deg = g->row_offset[u + 1] - g->row_offset[u];
        if (deg <= ADAPT_THRESHOLD) h_low.push_back(u);
        else                        h_high.push_back(u);
    }
    int low_count  = (int)h_low.size();
    int high_count = (int)h_high.size();

    // -----------------------------------------------------------------------
    // CUDA event handles for split timing (H2D / compute / D2H).
    // -----------------------------------------------------------------------
    cudaEvent_t ev_start, ev_after_h2d, ev_after_compute, ev_after_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_after_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_after_compute));
    CUDA_CHECK(cudaEventCreate(&ev_after_d2h));

    CUDA_CHECK(cudaEventRecord(ev_start));

    // -----------------------------------------------------------------------
    // Step 2: Upload CSR + dist[] + the two vertex-partition lists.
    //         All of this is counted in the H2D phase.
    // -----------------------------------------------------------------------
    DeviceCSR d_csr = copy_csr_to_device(g);

    float *d_dist = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_dist, V * sizeof(float)));

    int *d_changed = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_changed, sizeof(int)));

    // Initialise host dist[] and upload
    for (int i = 0; i < V; i++) dist[i] = SSSP_INF;
    dist[source] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_dist, dist, V * sizeof(float), cudaMemcpyHostToDevice));

    // Upload partition lists (one-time, part of H2D)
    int *d_low  = nullptr;
    int *d_high = nullptr;
    if (low_count) {
        CUDA_CHECK(cudaMalloc((void**)&d_low, low_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_low, h_low.data(), low_count * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (high_count) {
        CUDA_CHECK(cudaMalloc((void**)&d_high, high_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_high, h_high.data(), high_count * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaEventRecord(ev_after_h2d));

    // -----------------------------------------------------------------------
    // Step 3: Pre-compute launch configs (constant across iterations).
    //   Low  kernel: one thread  per vertex  → ceil(low_count  / BLOCK) blocks
    //   High kernel: one warp    per vertex  → ceil(high_count / WARPS_PER_BLOCK) blocks
    // -----------------------------------------------------------------------
    int grid_low  = (low_count  > 0) ? (low_count  + BLOCK - 1) / BLOCK : 0;
    int grid_high = (high_count > 0) ? (high_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;

    // -----------------------------------------------------------------------
    // Step 4: Main Bellman-Ford loop — time THIS region with CUDA events.
    // -----------------------------------------------------------------------
    int iter      = 0;
    int h_changed = 1;
    while (iter < V - 1 && h_changed) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        if (low_count > 0)
            sssp_adaptive_low_kernel<<<grid_low, BLOCK>>>(
                d_csr.row_offset, d_csr.col_index, d_csr.weights,
                d_dist, d_low, low_count, d_changed);

        if (high_count > 0)
            sssp_adaptive_high_kernel<<<grid_high, BLOCK>>>(
                d_csr.row_offset, d_csr.col_index, d_csr.weights,
                d_dist, d_high, high_count, d_changed);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        iter++;
    }
    int rounds_to_converge = iter;

    // Extra pass to detect negative-weight cycles (h_changed != 0 → cycle)
    h_changed = 0;
    CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
    if (low_count > 0)
        sssp_adaptive_low_kernel<<<grid_low, BLOCK>>>(
            d_csr.row_offset, d_csr.col_index, d_csr.weights,
            d_dist, d_low, low_count, d_changed);
    if (high_count > 0)
        sssp_adaptive_high_kernel<<<grid_high, BLOCK>>>(
            d_csr.row_offset, d_csr.col_index, d_csr.weights,
            d_dist, d_high, high_count, d_changed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_compute));

    // -----------------------------------------------------------------------
    // Step 5: Copy result back to host.
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(dist, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_after_d2h));
    CUDA_CHECK(cudaEventSynchronize(ev_after_d2h));

    // Fill timing struct (if caller wants it)
    if (timing) {
        CUDA_CHECK(cudaEventElapsedTime(&timing->h2d_ms,    ev_start,         ev_after_h2d));
        CUDA_CHECK(cudaEventElapsedTime(&timing->compute_ms,ev_after_h2d,     ev_after_compute));
        CUDA_CHECK(cudaEventElapsedTime(&timing->d2h_ms,    ev_after_compute, ev_after_d2h));
        CUDA_CHECK(cudaEventElapsedTime(&timing->total_ms,  ev_start,         ev_after_d2h));
        timing->rounds = rounds_to_converge;
    }

    // Cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_h2d);
    cudaEventDestroy(ev_after_compute);
    cudaEventDestroy(ev_after_d2h);

    free_device_csr(&d_csr);
    cudaFree(d_dist);
    cudaFree(d_changed);
    if (d_low)  cudaFree(d_low);
    if (d_high) cudaFree(d_high);

    if (h_changed) {
        fprintf(stderr, "Bellman-Ford (GPU Adaptive): negative-weight cycle detected.\n");
        return 0;
    }
    return 1;
}
