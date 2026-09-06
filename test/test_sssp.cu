#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "csr.h"
#include "sssp.h"
#include "timer.h"

/* Compares two distance arrays, printing up to 10 mismatches.
   Returns the mismatch count (0 = match). */
static int compare_arrays(const float *a, const float *b, int V,
                           const char *label_a, const char *label_b) {
    int mismatches = 0;
    for (int i = 0; i < V; i++) {
        if (fabsf(a[i] - b[i]) > 1e-3f) {
            if (mismatches < 10)
                fprintf(stderr, "  mismatch at vertex %d: %s=%f %s=%f\n",
                        i, label_a, a[i], label_b, b[i]);
            mismatches++;
        }
    }
    return mismatches;
}

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

    float *dist_c  = (float *)malloc(sizeof(float) * csr.V); /* CPU */
    float *dist_gv = (float *)malloc(sizeof(float) * csr.V); /* GPU thread-per-vertex */
    float *dist_ge = (float *)malloc(sizeof(float) * csr.V); /* GPU thread-per-edge */
    if (!dist_c || !dist_gv || !dist_ge) {
        fprintf(stderr, "out of memory\n");
        free(g.edges); free_csr(&csr);
        free(dist_c); free(dist_gv); free(dist_ge);
        return 1;
    }

    /* 3. Run CPU Bellman-Ford, timed. */
    Timer t_cpu;
    timer_start(&t_cpu);
    int ok_c = sssp_bellman_ford_cpu(&csr, source, dist_c);
    timer_stop(&t_cpu);
    double ms_cpu = timer_elapsed_ms(&t_cpu);

    /* 4. Run GPU Bellman-Ford, thread-per-vertex. GpuTiming captures the
          internal H2D/compute/D2H breakdown; ms_gv (wall-clock, includes
          any CPU-side call overhead) is kept too for reference. */
    GpuTiming tim_gv;
    memset(&tim_gv, 0, sizeof(tim_gv));
    Timer t_gv;
    timer_start(&t_gv);
    int ok_gv = sssp_bellman_ford_gpu(&csr, source, dist_gv, &tim_gv);
    timer_stop(&t_gv);
    double ms_gv = timer_elapsed_ms(&t_gv);

    /* 5. Run GPU Bellman-Ford, thread-per-edge. Same breakdown. */
    GpuTiming tim_ge;
    memset(&tim_ge, 0, sizeof(tim_ge));
    Timer t_ge;
    timer_start(&t_ge);
    int ok_ge = sssp_bellman_ford_gpu_t_per_e(&csr, source, dist_ge, &tim_ge);
    timer_stop(&t_ge);
    double ms_ge = timer_elapsed_ms(&t_ge);

    /* 6. Correctness checks: CPU vs each GPU variant, provided none of
          them hit a negative-weight cycle. */
    int mismatches_gv = 0, mismatches_ge = 0;
    int skip_compare = (!ok_c || !ok_gv || !ok_ge);

    if (skip_compare) {
        fprintf(stderr,
                "Negative-weight cycle detected (cpu ok=%d, gpu_t_per_v ok=%d, "
                "gpu_t_per_e ok=%d) - skipping compare\n", ok_c, ok_gv, ok_ge);
    } else {
        mismatches_gv = compare_arrays(dist_c, dist_gv, csr.V, "cpu", "gpu_t_per_v");
        mismatches_ge = compare_arrays(dist_c, dist_ge, csr.V, "cpu", "gpu_t_per_e");
    }

    /* 7. Report a single unified timing table: CPU total, then each GPU
          variant broken into H2D / compute / D2H / total, plus speedups. */
    printf("\n================= Timing Summary =================\n");
    printf("%-28s\n", "CPU Bellman-Ford");
    printf("  %-26s %10.4f ms\n", "total", ms_cpu);

    printf("\n%-28s\n", "GPU Bellman-Ford (thread-per-vertex)");
    printf("  %-26s %10.4f ms\n", "H2D transfer", tim_gv.h2d_ms);
    printf("  %-26s %10.4f ms  (%d rounds)\n", "compute", tim_gv.compute_ms, tim_gv.rounds);
    printf("  %-26s %10.4f ms\n", "D2H transfer", tim_gv.d2h_ms);
    printf("  %-26s %10.4f ms\n", "total (GPU-internal)", tim_gv.total_ms);
    printf("  %-26s %10.4f ms\n", "total (wall-clock)", ms_gv);

    printf("\n%-28s\n", "GPU Bellman-Ford (thread-per-edge)");
    printf("  %-26s %10.4f ms\n", "H2D transfer", tim_ge.h2d_ms);
    printf("  %-26s %10.4f ms  (%d rounds)\n", "compute", tim_ge.compute_ms, tim_ge.rounds);
    printf("  %-26s %10.4f ms\n", "D2H transfer", tim_ge.d2h_ms);
    printf("  %-26s %10.4f ms\n", "total (GPU-internal)", tim_ge.total_ms);
    printf("  %-26s %10.4f ms\n", "total (wall-clock)", ms_ge);

    printf("\n%-28s\n", "Speedups (CPU / GPU)");
    printf("  %-26s %10.2fx\n", "t/vertex, incl. transfer", ms_cpu / tim_gv.total_ms);
    printf("  %-26s %10.2fx\n", "t/vertex, compute only", ms_cpu / tim_gv.compute_ms);
    printf("  %-26s %10.2fx\n", "t/edge,   incl. transfer", ms_cpu / tim_ge.total_ms);
    printf("  %-26s %10.2fx\n", "t/edge,   compute only", ms_cpu / tim_ge.compute_ms);
    printf("  %-26s %10.2fx %s\n", "t/vertex vs t/edge (total)",
           tim_ge.total_ms / tim_gv.total_ms,
           (tim_gv.total_ms < tim_ge.total_ms) ? "(t/vertex faster)" : "(t/edge faster)");
    printf("====================================================\n");

    printf("\n--- Correctness ---\n");
    if (skip_compare) {
        printf("SKIPPED (negative-weight cycle on cpu=%d, gpu_t_per_v=%d, gpu_t_per_e=%d)\n",
               ok_c, ok_gv, ok_ge);
    } else {
        printf("t/vertex: %s (%d / %d vertices mismatched)\n",
               (mismatches_gv == 0) ? "PASS" : "FAIL", mismatches_gv, csr.V);
        printf("t/edge  : %s (%d / %d vertices mismatched)\n",
               (mismatches_ge == 0) ? "PASS" : "FAIL", mismatches_ge, csr.V);
    }

    int all_ok = ok_c && ok_gv && ok_ge &&
                 (skip_compare || (mismatches_gv == 0 && mismatches_ge == 0));

    free(g.edges);
    free_csr(&csr);
    free(dist_c);
    free(dist_gv);
    free(dist_ge);
    return all_ok ? 0 : 1;
}
