/*
 * csr_convert.c
 *
 * Standalone C program: reads a graph from a text file and converts it
 * to CSR (Compressed Sparse Row) format.
 *
 * Input file format:
 *   line 1:        <num_vertices> <num_edges>
 *   next E lines:  <u> <v> <w>      (0-indexed vertices, w = edge weight)
 *
 * CSR output (three arrays):
 *   row_offset[V+1] : row_offset[i] .. row_offset[i+1]-1 is the range of
 *                     edges (in col_index/weights) belonging to vertex i
 *   col_index[E]    : destination vertex of each edge
 *   weights[E]      : weight of each edge, parallel to col_index
 *
 * Usage:
 *   ./csr_convert <input_file> [--dump]
 *
 *   --dump   print the full row_offset / col_index / weights arrays
 *            (only use on small graphs)
 *
 * Build:
 *   gcc -O2 -std=c11 -o csr_convert csr_convert.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Graph structures -------------------------------------------------- */

typedef struct {
    int u, v;
    float w;
} Edge;

typedef struct {
    int V, E;
    Edge *edges;
} EdgeListGraph;

typedef struct {
    int V, E;
    int *row_offset;   /* size V+1 */
    int *col_index;    /* size E   */
    float *weights;    /* size E   */
} CSRGraph;

/* ---- Reading the input file -------------------------------------------- */

/* Reads the graph from `path`. Exits the program with an error message on
 * any malformed input (this keeps the example simple; a library version
 * would return an error code instead). */
EdgeListGraph read_graph_from_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: could not open input file '%s'\n", path);
        exit(1);
    }

    EdgeListGraph g;
    if (fscanf(fp, "%d %d", &g.V, &g.E) != 2) {
        fprintf(stderr, "Error: failed to read header (<num_vertices> <num_edges>) "
                        "from '%s'\n", path);
        fclose(fp);
        exit(1);
    }
    if (g.V < 0 || g.E < 0) {
        fprintf(stderr, "Error: header has negative vertex/edge count: %d %d\n",
                g.V, g.E);
        fclose(fp);
        exit(1);
    }

    g.edges = (Edge *)malloc(sizeof(Edge) * (size_t)g.E);
    if (g.E > 0 && !g.edges) {
        fprintf(stderr, "Error: out of memory allocating %d edges\n", g.E);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < g.E; i++) {
        int u, v;
        float w;
        if (fscanf(fp, "%d %d %f", &u, &v, &w) != 3) {
            fprintf(stderr, "Error: expected %d edge lines but failed to read "
                            "edge %d (only read %d successfully). Check the file "
                            "has '<u> <v> <w>' per line and matches the header count.\n",
                    g.E, i, i);
            fclose(fp);
            free(g.edges);
            exit(1);
        }
        if (u < 0 || u >= g.V || v < 0 || v >= g.V) {
            fprintf(stderr, "Error: edge %d has out-of-range endpoint(s): (%d, %d); "
                            "expected both in [0, %d)\n", i, u, v, g.V);
            fclose(fp);
            free(g.edges);
            exit(1);
        }
        g.edges[i].u = u;
        g.edges[i].v = v;
        g.edges[i].w = w;
    }

    fclose(fp);
    return g;
}

/* ---- Edge list -> CSR conversion (O(V + E)) ----------------------------- */

/* If `directed` is 0, every edge (u, v, w) is inserted as both (u, v, w)
 * and (v, u, w), so csr.E ends up being 2 * g.E. */
CSRGraph build_csr(const EdgeListGraph *g, int directed) {
    CSRGraph csr;
    csr.V = g->V;
    csr.E = directed ? g->E : g->E * 2;

    csr.row_offset = (int *)calloc((size_t)csr.V + 1, sizeof(int));
    csr.col_index  = (int *)malloc(sizeof(int) * (size_t)csr.E);
    csr.weights    = (float *)malloc(sizeof(float) * (size_t)csr.E);
    if (!csr.row_offset || (csr.E > 0 && (!csr.col_index || !csr.weights))) {
        fprintf(stderr, "Error: out of memory building CSR for V=%d E=%d\n",
                csr.V, csr.E);
        exit(1);
    }

    /* Pass 1: count out-degree of every vertex.
     * row_offset[i] temporarily holds the degree of vertex i. */
    for (int i = 0; i < g->E; i++) {
        csr.row_offset[g->edges[i].u]++;
        if (!directed) {
            csr.row_offset[g->edges[i].v]++;
        }
    }

    /* Prefix sum: degree counts -> starting offsets.
     * After this, row_offset[i] is the start index of vertex i's edges,
     * and row_offset[V] == E (sentinel). */
    int running = 0;
    for (int i = 0; i < csr.V; i++) {
        int degree = csr.row_offset[i];
        csr.row_offset[i] = running;
        running += degree;
    }
    csr.row_offset[csr.V] = running; /* == csr.E */

    /* Pass 2: place each edge into col_index/weights.
     * `cursor[i]` tracks how many of vertex i's edges have been placed so
     * far, starting from its row_offset. We use a separate scratch array
     * so row_offset itself is left intact for use afterwards. */
    int *cursor = (int *)malloc(sizeof(int) * (size_t)csr.V);
    if (!cursor) {
        fprintf(stderr, "Error: out of memory allocating cursor array\n");
        exit(1);
    }
    memcpy(cursor, csr.row_offset, sizeof(int) * (size_t)csr.V);

    for (int i = 0; i < g->E; i++) {
        int u = g->edges[i].u, v = g->edges[i].v;
        float w = g->edges[i].w;

        int pos = cursor[u]++;
        csr.col_index[pos] = v;
        csr.weights[pos] = w;

        if (!directed) {
            int pos_rev = cursor[v]++;
            csr.col_index[pos_rev] = u;
            csr.weights[pos_rev] = w;
        }
    }

    free(cursor);
    return csr;
}

/* ---- Basic structural validation ---------------------------------------- */

/* Returns 1 if all checks pass, 0 otherwise (prints the first problem found). */
int validate_csr(const CSRGraph *g) {
    if (g->row_offset[0] != 0) {
        fprintf(stderr, "Validation error: row_offset[0] must be 0, got %d\n",
                g->row_offset[0]);
        return 0;
    }
    if (g->row_offset[g->V] != g->E) {
        fprintf(stderr, "Validation error: row_offset[V] must equal E (%d), got %d\n",
                g->E, g->row_offset[g->V]);
        return 0;
    }
    for (int i = 0; i < g->V; i++) {
        if (g->row_offset[i] > g->row_offset[i + 1]) {
            fprintf(stderr, "Validation error: row_offset not monotonic at index %d\n", i);
            return 0;
        }
    }
    for (int i = 0; i < g->E; i++) {
        if (g->col_index[i] < 0 || g->col_index[i] >= g->V) {
            fprintf(stderr, "Validation error: col_index[%d] = %d out of range [0, %d)\n",
                    i, g->col_index[i], g->V);
            return 0;
        }
    }
    return 1;
}

/* ---- Debug printing ------------------------------------------------------ */

void dump_csr(const CSRGraph *g) {
    printf("\nrow_offset (%d): ", g->V + 1);
    for (int i = 0; i <= g->V; i++) printf("%d ", g->row_offset[i]);
    printf("\ncol_index  (%d): ", g->E);
    for (int i = 0; i < g->E; i++) printf("%d ", g->col_index[i]);
    printf("\nweights    (%d): ", g->E);
    for (int i = 0; i < g->E; i++) printf("%.3f ", g->weights[i]);
    printf("\n");
}

void free_csr(CSRGraph *g) {
    free(g->row_offset);
    free(g->col_index);
    free(g->weights);
}

/* ---- main ----------------------------------------------------------------- */

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