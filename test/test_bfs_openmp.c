#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csr.h"
#include "bfs.h"

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"Usage: %s <graph> [source] [--undirected]\n",argv[0]); return 1; }
    const char *path=argv[1]; int source=0, directed=1;
    for(int i=2;i<argc;i++){ if(!strcmp(argv[i],"--undirected")) directed=0; else source=atoi(argv[i]); }
    EdgeListGraph g=read_graph_from_file(path);
    CSRGraph csr=build_csr(&g,directed);
    if(!validate_csr(&csr)){ fprintf(stderr,"CSR invalid\n"); return 1; }
    printf("Graph: V=%d E=%d source=%d\n",csr.V,csr.E,source);
    int *a=(int*)malloc(sizeof(int)*csr.V), *b=(int*)malloc(sizeof(int)*csr.V);
    bfs_cpu(&csr,source,a);
    bfs_openmp(&csr,source,b);
    int m=0; for(int i=0;i<csr.V;i++) if(a[i]!=b[i]){ if(m<10) fprintf(stderr,"  mismatch v%d: cpu=%d omp=%d\n",i,a[i],b[i]); m++; }
    printf("OpenMP vs CPU: %s (%d mismatches)\n", m==0?"PASS":"FAIL", m);
    free(a); free(b); free(g.edges); free_csr(&csr);
    return m==0?0:1;
}