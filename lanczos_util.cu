#include "lanczos_util.h"
#include <device_launch_parameters.h>

__global__ __forceinline__
void _print_vec_f(float* vec, size_t n)
{
    for (int i = 0; i < n; i++) printf("%f ", vec[i]);
    printf("\n");
}

void print_vec_f_dev(float* vec, size_t n)
{
    _print_vec_f << <1, 1 >> > (vec, n);
    CUDA_CALL(cudaDeviceSynchronize());
}

__global__ __forceinline__
void _scale_vector_f(float* vec, float sca, float offset, size_t n)
{
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId < n)
    {
        vec[threadId] = vec[threadId] * sca + offset;
    }
}

void scale_vec(float* vec, float sca, float offset, size_t n)
{
    int block_num = (n - 1) / (32 * 32) + 1;
    dim3 dimBlockSize(32, 32, 1);
    _scale_vector_f << <block_num, dimBlockSize >> > (vec, sca, offset, n);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

void normalize(float* vec, size_t n, cublasHandle_t handle)
{
    float _norm;
    CUBLAS_CALL(cublasSnrm2(handle, n, vec, 1, &_norm));
    if (_norm < EPS)
        return;
    _norm = 1. / _norm;
    CUBLAS_CALL(cublasSscal(handle, n, &_norm, vec, 1));
}

void schmidt_orth(float* vs, float* xn, size_t n, size_t v_cnt, cublasHandle_t handle)
{
    //printf("schmidt\n");
    //print_vec_f_dev(vs, n * v_cnt);
    //print_vec_f_dev(xn, n);
    if (v_cnt >= n) {
        CUDA_CALL(cudaMemset(xn, 0, n * sizeof(float)));
        return;
    }
    float alpha = -1.0;
    float beta = 1.0;
    float* dots;
    float* sum;
    CUDA_CALL(cudaMalloc(&dots, v_cnt * sizeof(float)));
    CUDA_CALL(cudaMalloc(&sum, n * sizeof(float)));
    CUDA_CALL(cudaMemset(dots, 0, v_cnt * sizeof(float)));
    CUDA_CALL(cudaMemset(sum, 0, n * sizeof(float)));
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_T, n, v_cnt, &alpha, vs, n, xn, 1, &beta, dots, 1));
    //print_vec_f_dev(dots, v_cnt);
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, n, v_cnt, &beta, vs, n, dots, 1, &beta, sum, 1));
    CUBLAS_CALL(cublasSaxpy(handle, n, &beta, sum, 1, xn, 1));
}

/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for the original matrix.
 */
 //void eigenvector(
 //    const float* alpha,
 //    const float* beta,
 //    const float* u,
 //    const float* evs,
 //    const size_t u_size_n,
 //    const size_t u_size_m)
 //{
 //    std::vector<LT> eigvec(n, 0.0);
 //
 //    auto cv = tridiagonal::tridiagonal_eigenvector(alpha, beta, ev);
 //
 //    for (size_t k = m; k-- > 0;) {
 //        for (size_t i = 0; i < n; ++i) {
 //            eigvec[i] += cv[k] * u[k][i];
 //        }
 //    }
 //
 //    util::normalize(eigvec);
 //
 //    return eigvec;
 //}

