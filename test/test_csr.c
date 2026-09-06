#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/csr.h"

//main fnc
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [--undirected] [--dump]\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    int directed = 1;
    int dump = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--undirected") == 0) {
            directed = 0;
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump = 1;
        } else {
            fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
            return 1;
        }
    }

    EdgeListGraph g = read_graph_from_file(input_path);
    printf("Read graph: V=%d E=%d from %s\n", g.V, g.E, input_path);

    CSRGraph csr = build_csr(&g, directed);

    if (!validate_csr(&csr)) {
        fprintf(stderr, "CSR validation FAILED\n");
        free(g.edges);
        free_csr(&csr);
        return 1;
    }

    printf("CSR built and validated OK\n");
    printf("  directed mode : %s\n", directed ? "directed" : "undirected (edges doubled)");
    printf("  V             : %d\n", csr.V);
    printf("  E (CSR)       : %d\n", csr.E);

    if (csr.V > 0) {
        int min_deg = csr.row_offset[1] - csr.row_offset[0];
        int max_deg = min_deg;
        for (int i = 1; i < csr.V; i++) {
            int d = csr.row_offset[i + 1] - csr.row_offset[i];
            if (d < min_deg) min_deg = d;
            if (d > max_deg) max_deg = d;
        }
        printf("  min out-deg   : %d\n", min_deg);
        printf("  max out-deg   : %d\n", max_deg);
        printf("  avg out-deg   : %.3f\n", (double)csr.E / csr.V);
    }

    if (dump) {
        dump_csr(&csr);
    }

    free(g.edges);
    free_csr(&csr);
    return 0;
}