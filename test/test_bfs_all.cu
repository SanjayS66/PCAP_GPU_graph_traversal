#include <stdio.h>
#include <stdlib.h>

#include "csr.h"
#include "bfs.h"

static int compare(
    const int *a,
    const int *b,
    int V,
    const char *name)
{
    int mismatches = 0;

    for (int i = 0; i < V; i++)
    {
        if (a[i] != b[i])
        {
            if (mismatches < 10)
            {
                printf(
                    "Mismatch in %s at vertex %d: "
                    "CPU=%d GPU=%d\n",
                    name,
                    i,
                    a[i],
                    b[i]);
            }

            mismatches++;
        }
    }

    if (mismatches == 0)
    {
        printf("%s: PASS\n", name);
    }
    else
    {
        printf("%s: FAIL (%d mismatches)\n",
               name,
               mismatches);
    }

    return mismatches;
}


int main()
{
    const char *filename =
        "graphs/sample_graph.txt";

    int source = 0;

    EdgeListGraph graph =
        read_graph_from_file(filename);

    CSRGraph csr =
        build_csr(&graph, 0);

    printf("Graph: V=%d E=%d\n\n",
           csr.V,
           csr.E);

int *cpu =
    (int *)malloc(sizeof(int) * csr.V);

int *gpu_frontier =
    (int *)malloc(sizeof(int) * csr.V);

int *gpu_edge =
    (int *)malloc(sizeof(int) * csr.V);

if (!cpu || !gpu_frontier || !gpu_edge)
{
    printf("Memory allocation failed\n");
    return 1;
}

    printf("Running CPU BFS...\n");
    bfs_cpu(&csr, source, cpu);


    printf("Running GPU frontier BFS...\n");
    bfs_gpu(&csr, source, gpu_frontier);


    printf("Running GPU thread-per-edge BFS...\n");
    bfs_gpu_t_per_e(&csr, source, gpu_edge);


    printf("\n===== Correctness =====\n");

    int errors = 0;

    errors += compare(
        cpu,
        gpu_frontier,
        csr.V,
        "GPU frontier BFS");

    errors += compare(
        cpu,
        gpu_edge,
        csr.V,
        "GPU thread-per-edge BFS");


    printf("\n===== Distances =====\n");

    for (int i = 0; i < csr.V; i++)
    {
        printf(
            "Vertex %d: CPU=%d "
            "Frontier=%d "
            "Edge=%d\n",
            i,
            cpu[i],
            gpu_frontier[i],
            gpu_edge[i]);
    }


    free(cpu);
    free(gpu_frontier);
    free(gpu_edge);

    free_csr(&csr);
    free(graph.edges);

    return errors == 0 ? 0 : 1;
}
