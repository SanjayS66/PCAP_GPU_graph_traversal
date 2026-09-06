//cuda_utils.h

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <cstdio>
#include <cstdlib>
#include "csr.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
struct DeviceCSR {    //deviceCSR differnet because it has to be stored in the GPU memory
    int V, E;
    int* row_offset;
    int* col_index;
    float* weights;
};

#ifdef __cplusplus
extern "C" {
#endif

DeviceCSR copy_csr_to_device(const CSRGraph* h_csr);    //copy CPU memory to GPU memory
void free_device_csr(DeviceCSR* d_csr);   //Free GPU memory

#ifdef __cplusplus
}
#endif

// Device-only function — not extern "C", only usable inside .cu files
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }

    return __int_as_float(old);
}

#endif