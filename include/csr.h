#ifndef CSR_H
#define CSR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int u, v;
    float w;
} Edge;

//intermediate type storing total number of edges and vertices as well as array of all the edges
typedef struct {
    int V, E;
    Edge *edges;
} EdgeListGraph;

// output for the final csr
typedef struct {
    int V, E;
    int *row_offset;   /* (array of idx of first instance of zero)size V+1 */
    int *col_index;    /* stores destination of each edges(size E)   */
    float *weights;    /* (weight of E)size E   */
} CSRGraph;

// Read the input file 
EdgeListGraph read_graph_from_file(const char *path);

// Edge list to CSR conversion
CSRGraph build_csr(const EdgeListGraph *g, int directed);

// check for errors in the constructed csr
int validate_csr(const CSRGraph *g);

//debug print
void dump_csr(const CSRGraph *g);

void free_csr(CSRGraph *g);

#ifdef __cplusplus
}
#endif

#endif /* CSR_H */