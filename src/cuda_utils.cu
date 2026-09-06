#include "cuda_utils.h"

DeviceCSR copy_csr_to_device(const CSRGraph* h_csr){
    //take the host CSR and copy into the GPU memory
    DeviceCSR d_csr;
    d_csr.V = h_csr->V;
    d_csr.E = h_csr->E;

    size_t row_offset_bytes = (h_csr->V + 1) * sizeof(int);
    size_t col_idx_bytes = (h_csr->E) * sizeof(int);
    size_t weights_bytes = (h_csr->E) * sizeof(float);

    // just allocate the required memory
    CUDA_CHECK(cudaMalloc(&d_csr.row_offset, row_offset_bytes));
    CUDA_CHECK(cudaMalloc(&d_csr.col_index,  col_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_csr.weights,    weights_bytes));

    // Copy actual array contents from host to device
    CUDA_CHECK(cudaMemcpy(d_csr.row_offset, h_csr->row_offset,
                           row_offset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.col_index, h_csr->col_index,
                           col_idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.weights, h_csr->weights,
                           weights_bytes, cudaMemcpyHostToDevice));

    return d_csr;
}

void free_device_csr(DeviceCSR* d_csr){
    // release each device allocation made in copy_csr_to_device
    cudaFree(d_csr->row_offset);
    cudaFree(d_csr->col_index);
    cudaFree(d_csr->weights);
}

DeviceCSR_t_per_e copy_csr_to_device_t_per_e(const CSRGraph* h_csr) {
    DeviceCSR_t_per_e d;
    d.V = h_csr->V;
    d.E = h_csr->E;

    /* --- Build src_vertex on the host first (O(V+E), done once) ---
       For each vertex u, every edge index k in [row_offset[u], row_offset[u+1])
       belongs to u. This just "expands" row_offset back into a flat array. */
    int* h_src_vertex = (int*)malloc(sizeof(int) * (size_t)d.E);
    if (d.E > 0 && !h_src_vertex) {
        fprintf(stderr, "Error: out of memory building src_vertex (E=%d)\n", d.E);
        exit(EXIT_FAILURE);
    }
    for (int u = 0; u < h_csr->V; u++) {
        for (int k = h_csr->row_offset[u]; k < h_csr->row_offset[u + 1]; k++) {
            h_src_vertex[k] = u;
        }
    }

    /* --- Allocate device memory --- */
    CUDA_CHECK(cudaMalloc((void**)&d.row_offset, (d.V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d.col_index,  d.E * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d.weights,    d.E * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d.src_vertex, d.E * sizeof(int)));

    /* --- Copy host -> device --- */
    CUDA_CHECK(cudaMemcpy(d.row_offset, h_csr->row_offset,
                           (d.V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.col_index, h_csr->col_index,
                           d.E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.weights, h_csr->weights,
                           d.E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.src_vertex, h_src_vertex,
                           d.E * sizeof(int), cudaMemcpyHostToDevice));

    free(h_src_vertex);
    return d;
}

void free_device_csr_t_per_e(DeviceCSR_t_per_e* d_csr) {
    cudaFree(d_csr->row_offset);
    cudaFree(d_csr->col_index);
    cudaFree(d_csr->weights);
    cudaFree(d_csr->src_vertex);
}
