#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csr.h"
#include "sssp.h"

#ifdef _OPENMP
#include <omp.h>
#define WALLTIME() omp_get_wtime()
#else
#include <time.h>
#define WALLTIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [source] [--undirected] [--omp] [--save <file>]\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *save_path  = NULL;
    int source = 0, directed = 1, use_omp = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0)      directed = 0;
        else if (strcmp(argv[i], "--omp") == 0)        use_omp = 1;
        else if (strcmp(argv[i], "--save") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--save needs a filename\n"); return 1; }
            save_path = argv[++i];
        }
        else source = atoi(argv[i]);                   /* a bare number = source vertex */
    }

    EdgeListGraph g = read_graph_from_file(input_path);
    CSRGraph csr = build_csr(&g, directed);
    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(g.edges); free_csr(&csr); return 1;
    }
    if (source < 0 || source >= csr.V) {
        fprintf(stderr, "Source %d out of range [0, %d)\n", source, csr.V);
        free(g.edges); free_csr(&csr); return 1;
    }

    printf("Graph: V=%d E=%d | source=%d | %s | mode=%s\n",
           csr.V, csr.E, source, directed ? "directed" : "undirected",
           use_omp ? "OpenMP" : "sequential");
#ifdef _OPENMP
    if (use_omp) printf("OpenMP threads available: %d\n", omp_get_max_threads());
#endif

    float *dist = (float *)malloc(sizeof(float) * csr.V);
    if (!dist) { fprintf(stderr, "out of memory\n"); return 1; }

    double t0 = WALLTIME();
    int ok = use_omp ? sssp_bellman_ford_openmp(&csr, source, dist)
                     : sssp_bellman_ford_cpu(&csr, source, dist);
    double t1 = WALLTIME();

    if (ok) {
        int reachable = 0;
        for (int i = 0; i < csr.V; i++) if (dist[i] != SSSP_INF) reachable++;
        printf("Reachable from %d: %d / %d vertices\n", source, reachable, csr.V);
        printf("Solve time: %.4f s\n", t1 - t0);

        int limit = csr.V < 10 ? csr.V : 10;
        printf("First %d distances:\n", limit);
        for (int i = 0; i < limit; i++) {
            if (dist[i] == SSSP_INF) printf("  dist[%d] = INF (unreachable)\n", i);
            else                     printf("  dist[%d] = %.3f\n", i, dist[i]);
        }

        if (save_path) {
            FILE *fp = fopen(save_path, "w");
            if (fp) {
                fprintf(fp, "%d\n", csr.V);
                for (int i = 0; i < csr.V; i++) fprintf(fp, "%.6f\n", dist[i]);
                fclose(fp);
                printf("Saved %d distances to %s\n", csr.V, save_path);
            } else fprintf(stderr, "could not open %s for writing\n", save_path);
        }
    }

    free(dist); free(g.edges); free_csr(&csr);
    return ok ? 0 : 1;
}