
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csr.h"

/* ---- Graph structures -------------------------------------------------- */

// Read the input file 
EdgeListGraph read_graph_from_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: could not open input file '%s'\n", path);
        exit(1);
    }

    EdgeListGraph g;
    //No V & E at top of file 
    if (fscanf(fp, "%d %d", &g.V, &g.E) != 2) {
        fprintf(stderr, "Error: failed to read header (<num_vertices> <num_edges>) "
                        "from '%s'\n", path);
        fclose(fp);
        exit(1);
    }

    //negetive V & E values
    if (g.V < 0 || g.E < 0) {
        fprintf(stderr, "Error: header has negative vertex/edge count: %d %d\n",
                g.V, g.E);
        fclose(fp);
        exit(1);
    }

    //allocate memery
    g.edges = (Edge *)malloc(sizeof(Edge) * (size_t)g.E);  //size_t : An unsigned integer for memory sizes.
    if (g.E > 0 && !g.edges) {
        fprintf(stderr, "Error: out of memory allocating %d edges\n", g.E);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < g.E; i++) {
        int u, v;
        float w;
        //error in the graph input file format
        if (fscanf(fp, "%d %d %f", &u, &v, &w) != 3) {
            fprintf(stderr, "Error: expected %d edge lines but failed to read "
                            "edge %d (only read %d successfully). Check the file "
                            "has '<u> <v> <w>' per line and matches the header count.\n",
                    g.E, i, i);
            fclose(fp);
            free(g.edges);
            exit(1);
        }
        //invalid values of u and v
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

// Edge list to CSR conversion
// CSR has the 3 arrays : (1) row_offset => number of edges from vertex[i]  
// (2)weights => weight of edge[i]  
// (3)col_index => neighbours of specific vertices/destiantion vertices of all edges 

//  We calculate the out-degree of every vertex and store it in row_offset.
//  Then we replace each element of row_offset with the running sum of the
//  elements before it, so that the degree of vertex i can be
//  calculated as row_offset[i+1] - row_offset[i].
// 
//  Then we make a copy of row_offset and name it cursor.
// 
//  To calculate the destination (col_index) and weight (weights), we
//  iterate through the edges array, and for each edge (u, v, w) we place
//  v and w into csr.col_index and csr.weights at index cursor[u], then
//  increment cursor[u]. After this, all 3 CSR arrays are ready.
CSRGraph build_csr(const EdgeListGraph *g, int directed) {
    CSRGraph csr;
    csr.V = g->V;
    csr.E = directed ? g->E : g->E * 2;

    //row_offset given V+1 size becuase we find the degree of a vertex by row_offset[i+1] - row_offset[i] and hence to prevent overflow when calculating for V vertex
    csr.row_offset = (int *)calloc((size_t)csr.V + 1, sizeof(int));
    csr.col_index  = (int *)malloc(sizeof(int) * (size_t)csr.E);
    csr.weights    = (float *)malloc(sizeof(float) * (size_t)csr.E);
    if (!csr.row_offset || (csr.E > 0 && (!csr.col_index || !csr.weights))) {
        fprintf(stderr, "Error: out of memory building CSR for V=%d E=%d\n",
                csr.V, csr.E);
        exit(1);
    }

    // count out-degree of every vertex.
    // row_offset[i] temporarily holds the degree of vertex i. 
    for (int i = 0; i < g->E; i++) {
        csr.row_offset[g->edges[i].u]++;
        if (!directed) {
            csr.row_offset[g->edges[i].v]++;
        }
    }

    // Keeping a running sum of the degrees of previous vertices, and in every loop assigning row_offset[i] to that running sum, 
    // which becomes the row_offset for that vertex.
    int running = 0;
    for (int i = 0; i < csr.V; i++) {
        int degree = csr.row_offset[i];
        csr.row_offset[i] = running;
        running += degree;
    }
    csr.row_offset[csr.V] = running;
    //row_offset values correctly 

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



// check for errors in the constructed csr
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


//debug print
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

