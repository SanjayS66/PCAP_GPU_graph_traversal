#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include "csr.h"
#include "sssp.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file> [source] [--undirected]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int source = 0, directed = 1;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) directed = 0;
        else source = atoi(argv[i]);
    }

    EdgeListGraph g = read_graph_from_file(path);
    CSRGraph csr = build_csr(&g, directed);
    if (!validate_csr(&csr)) { fprintf(stderr, "CSR invalid\n"); return 1; }
    printf("Graph: V=%d E=%d  source=%d\n", csr.V, csr.E, source);

    float *dist_cpu  = (float*)malloc(sizeof(float) * csr.V);
    float *dist_warp = (float*)malloc(sizeof(float) * csr.V);

    clock_t c0 = clock();
    int ok_cpu = sssp_bellman_ford_cpu(&csr, source, dist_cpu);
    double ms_cpu = 1000.0 * (clock() - c0) / CLOCKS_PER_SEC;

    GpuTiming tim; memset(&tim, 0, sizeof(tim));
    int ok_gpu = sssp_bellman_ford_gpu_warp_per_v(&csr, source, dist_warp, &tim);

    /* Compare */
    int mismatches = 0;
    for (int i = 0; i < csr.V; i++) {
        if (fabsf(dist_cpu[i] - dist_warp[i]) > 1e-3f) {
            if (mismatches < 10)
                fprintf(stderr, "  mismatch v%d: cpu=%f warp=%f\n", i, dist_cpu[i], dist_warp[i]);
            mismatches++;
        }
    }

    printf("\nCPU total        : %.4f ms\n", ms_cpu);
    printf("GPU warp compute : %.4f ms  (%d rounds)\n", tim.compute_ms, tim.rounds);
    printf("GPU warp total   : %.4f ms  (H2D %.3f + compute %.3f + D2H %.3f)\n",
           tim.total_ms, tim.h2d_ms, tim.compute_ms, tim.d2h_ms);
    if (tim.compute_ms > 0)
        printf("Speedup (compute): %.2fx\n", ms_cpu / tim.compute_ms);

    printf("\nRESULT: %s (%d mismatches, cpu_ok=%d gpu_ok=%d)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches, ok_cpu, ok_gpu);

    free(dist_cpu); free(dist_warp); free(g.edges); free_csr(&csr);
    return mismatches == 0 ? 0 : 1;
}