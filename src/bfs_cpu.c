#include <stdio.h>
#include <stdlib.h>
#include "csr.h"
#include "bfs.h"

void bfs_cpu(const CSRGraph *graph, int source, int *distance)
{
    if (graph == NULL || distance == NULL)
        return;

    if (source < 0 || source >= graph->V)
        return;

    int V = graph->V;

    int *queue = (int *)malloc(sizeof(int) * V);

    if (queue == NULL)
    {
        fprintf(stderr, "Error: could not allocate BFS queue\n");
        exit(1);
    }

    for (int i = 0; i < V; i++)
        distance[i] = -1;

    int front = 0;
    int rear = 0;

    distance[source] = 0;
    queue[rear++] = source;

    while (front < rear)
    {
        int vertex = queue[front++];

        /*
         * CSR stores neighbours of vertex in:
         *
         * row_offset[vertex]
         *        to
         * row_offset[vertex + 1] - 1
         */
        int start = graph->row_offset[vertex];
        int end   = graph->row_offset[vertex + 1];

        for (int i = start; i < end; i++)
        {
            int neighbour = graph->col_index[i];

            if (distance[neighbour] == -1)
            {
                distance[neighbour] = distance[vertex] + 1;

                queue[rear++] = neighbour;
            }
        }
    }

    free(queue);
}