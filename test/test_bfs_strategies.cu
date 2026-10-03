#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "csr.h"
#include "bfs.h"

static int compare(const int *a, const int *b, int V, const char *name) {
    int m = 0;
    for (int i = 0; i < V; i++)
        if (a[i] != b[i]) {
            if (m < 10) fprintf(stderr, "  %s mismatch v%d: cpu=%d gpu=%d\n", name, i, a[i], b[i]);
            m++;
        }
    printf("%-18s : %s (%d mismatches)\n", name, m == 0 ? "PASS" : "FAIL", m);
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <graph> [source] [--undirected]\n", argv[0]); return 1; }
    const char *path = argv[1];
    int source = 0, directed = 1;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) directed = 0;
        else source = atoi(argv[i]);
    }

    EdgeListGraph g = read_graph_from_file(path);
    CSRGraph csr = build_csr(&g, directed);
    if (!validate_csr(&csr)) { fprintf(stderr, "CSR invalid\n"); return 1; }
    printf("Graph: V=%d E=%d  source=%d\n\n", csr.V, csr.E, source);

    int *d_cpu  = (int*)malloc(sizeof(int)*csr.V);
    int *d_omp  = (int*)malloc(sizeof(int)*csr.V);
    int *d_warp = (int*)malloc(sizeof(int)*csr.V);
    int *d_edge = (int*)malloc(sizeof(int)*csr.V);

    bfs_cpu(&csr, source, d_cpu);
    bfs_openmp(&csr, source, d_omp);
    GpuTiming tw, te; memset(&tw,0,sizeof(tw)); memset(&te,0,sizeof(te));
    bfs_gpu_warp_per_v(&csr, source, d_warp, &tw);
    bfs_gpu_edge_based(&csr, source, d_edge, &te);

    printf("--- Correctness vs CPU BFS ---\n");
    compare(d_cpu, d_omp,  csr.V, "OpenMP");
    compare(d_cpu, d_warp, csr.V, "GPU warp/vertex");
    compare(d_cpu, d_edge, csr.V, "GPU edge-based");

    printf("\n--- GPU compute time ---\n");
    printf("warp/vertex : %.3f ms (%d levels)\n", tw.compute_ms, tw.rounds);
    printf("edge-based  : %.3f ms (%d levels)\n", te.compute_ms, te.rounds);

    free(d_cpu); free(d_omp); free(d_warp); free(d_edge);
    free(g.edges); free_csr(&csr);
    return 0;
}