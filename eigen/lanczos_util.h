#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cstdio>

#define EPS 1e-6

#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}

#define CURAND_CALL(x) if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}

#define CUBLAS_CALL(x) if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n", x, __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}

void print_vec_f_dev(float* vec, size_t n);

void scale_vec(float* vec, float sca, float offset, size_t n);

void normalize(float* vec, size_t n, cublasHandle_t handle);
void schmidt_orth(float* vs, float* xn, size_t n, size_t v_cnt, cublasHandle_t handle);

/**
 * @brief Returns 1-norm of given array.
 */
template <typename T>
inline T l1_norm_host(const T * vec, size_t size) {
    T norm = 0.0;
    for (int i = 0; i < size; i++) {
        norm += std::abs(vec[i]);
    }
    return norm;
}
