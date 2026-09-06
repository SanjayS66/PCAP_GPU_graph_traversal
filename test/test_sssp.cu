#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "csr.h"
#include "sssp.h"
#include "timer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [source] [--undirected]\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    int source = 0, directed = 1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) directed = 0;
        else                                      source = atoi(argv[i]); /* bare number = source */
    }

    /* 1. Read edge list from file. */
    EdgeListGraph g = read_graph_from_file(input_path);
    printf("Read graph: V=%d E=%d from %s\n", g.V, g.E, input_path);

    /* 2. Build + validate CSR. */
    CSRGraph csr = build_csr(&g, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(g.edges);
        free_csr(&csr);
        return 1;
    }
    printf("CSR built and validated OK (V=%d, E=%d)\n", csr.V, csr.E);

    if (source < 0 || source >= csr.V) {
        fprintf(stderr, "Source %d out of range [0, %d)\n", source, csr.V);
        free(g.edges);
        free_csr(&csr);
        return 1;
    }

    float *dist_c = (float *)malloc(sizeof(float) * csr.V);
    float *dist_g = (float *)malloc(sizeof(float) * csr.V);
    if (!dist_c || !dist_g) {
        fprintf(stderr, "out of memory\n");
        free(g.edges); free_csr(&csr); free(dist_c); free(dist_g);
        return 1;
    }

    /* 3. Run CPU Bellman-Ford, timed. */
    Timer t_cpu;
    timer_start(&t_cpu);
    int ok_c = sssp_bellman_ford_cpu(&csr, source, dist_c);
    timer_stop(&t_cpu);

    /* 4. Run GPU Bellman-Ford, timed. */
    Timer t_gpu;
    timer_start(&t_gpu);
    int ok_g = sssp_bellman_ford_gpu(&csr, source, dist_g);
    timer_stop(&t_gpu);

    /* 5. Correctness check: CPU and GPU distances should agree, provided
          neither hit a negative-weight cycle. */
    int correct = 0;
    int mismatches = 0;
    if (!ok_c || !ok_g) {
        fprintf(stderr, "Negative-weight cycle detected (cpu ok=%d, gpu ok=%d) - skipping compare\n",
                ok_c, ok_g);
    } else {
        for (int i = 0; i < csr.V; i++) {
            if (fabsf(dist_c[i] - dist_g[i]) > 1e-3f) {
                if (mismatches < 10)
                    fprintf(stderr, "  mismatch at vertex %d: cpu=%f gpu=%f\n",
                            i, dist_c[i], dist_g[i]);
                mismatches++;
            }
        }
        correct = (mismatches == 0);
    }

    /* 6. Report timing + correctness. */
    printf("\n--- Timing ---\n");
    printf("CPU Bellman-Ford : %.4f ms\n", timer_elapsed_ms(&t_cpu));
    printf("GPU Bellman-Ford : %.4f ms\n", timer_elapsed_ms(&t_gpu));
    printf("Speedup (CPU/GPU): %.2fx\n", timer_elapsed_ms(&t_cpu) / timer_elapsed_ms(&t_gpu));

    printf("\n--- Correctness ---\n");
    if (!ok_c || !ok_g) {
        printf("SKIPPED (negative-weight cycle on cpu=%d and/or gpu=%d)\n", ok_c, ok_g);
    } else if (correct) {
        printf("PASS: CPU and GPU results match (V=%d)\n", csr.V);
    } else {
        printf("FAIL: %d / %d vertices mismatched\n", mismatches, csr.V);
    }

    free(g.edges);
    free_csr(&csr);
    free(dist_c);
    free(dist_g);
    return (ok_c && ok_g && correct) ? 0 : 1;
}