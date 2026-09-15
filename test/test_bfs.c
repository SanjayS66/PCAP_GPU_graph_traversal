#include <stdio.h>
#include <stdlib.h>

#include "csr.h"
#include "bfs.h"

int main()
{
    const char *filename = "graphs/sample_graph.txt";

    /* Read graph */
    EdgeListGraph graph = read_graph_from_file(filename);

    /* Build CSR as undirected graph */
    CSRGraph csr = build_csr(&graph, 0);

    /* Allocate distance array */
    int *distance = malloc(sizeof(int) * csr.V);

    if (distance == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* Run BFS from vertex 0 */
    bfs_cpu(&csr, 0, distance);

    /* Print results */
    printf("BFS starting from vertex 0:\n");

    for (int i = 0; i < csr.V; i++)
    {
        printf("Vertex %d -> distance %d\n", i, distance[i]);
    }

    free(distance);

    free(csr.row_offset);
    free(csr.col_index);
    free(csr.weights);

    free(graph.edges);

    return 0;
}