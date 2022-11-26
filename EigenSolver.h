#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "lanczos_util.h"

/**
 * @brief Returns the significant decimal digits of type T.
 *
 */
template <typename T>
inline constexpr int sig_decimal_digit() {
    return (int)(std::numeric_limits<T>::digits *
        log10(std::numeric_limits<T>::radix));
}


template <typename T>
inline constexpr T minimum_effective_decimal() {
    return pow(10, -sig_decimal_digit<T>());
}

template <typename T>
struct TridiagonalEigen
{
private:
    /**
    * @brief Finds the number of eigenvalues of given tridiagonal matrix smaller than `c`.
    *
    * Algorithm from
    * Peter Arbenz et al. / "High Performance Algorithms for Structured Matrix Problems" /
    * Nova Science Publishers, Inc.
    */
    static inline size_t num_of_eigs_smaller_than(
        T c,
        const T* alpha,
        const T* beta,
        size_t n)
    {
        T q_i = alpha[0] - c;
        size_t count = 0;
        if (q_i < 0) ++count;
        for (size_t i = 1; i < n; ++i) {
            q_i = alpha[i] - c - beta[i - 1] * beta[i - 1] / q_i;
            if (q_i < 0) ++count;
            if (q_i == 0) q_i = minimum_effective_decimal<T>();
        }
        return count;
    }

    /**
     * @brief Computes the upper bound of the absolute value of eigenvalues by Gerschgorin theorem.
     *
     * This routine gives a rough upper bound,
     * but it is sufficient because the bisection routine using
     * the upper bound converges exponentially.
     */
    static inline T tridiagonal_eigen_limit(
        const T* alpha,
        const T* beta,
        size_t n)
    {
        T r = l1_norm_host(alpha, n);
        r += 2 * l1_norm_host(beta, n);
        return r;
    }

public:
     /**
     * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
     */
    static inline T find_mth_eigenvalue(
        const T* alpha,
        const T* beta,
        const size_t m,
        const size_t n,
        const T eps) 
    {
        T mid;
        T pmid = std::numeric_limits<T>::max();
        T r = tridiagonal_eigen_limit(alpha, beta, n);
        T lower = -r;
        T upper = r;

        while (upper - lower > std::min(std::abs(lower), std::abs(upper)) * eps) {
            mid = (lower + upper) / 2.0;
            if (num_of_eigs_smaller_than(mid, alpha, beta, n) >= m + 1) upper = mid;
            else lower = mid;
            if (mid == pmid) break;  // This avoids an infinite loop due to zero matrix
            pmid = mid;
        }
        return lower;
    }
    /**
     * @brief Computes an eigenvector corresponding to given eigenvalue for the original matrix.
     */
    static inline void tridiagonal_eigenvector(const T* main_diag, const T* sub_diag, const T ev, T* e_vec, size_t n) {
        e_vec[n - 1] = 1.0;
        e_vec[n - 2] = ((ev - main_diag[n - 1]) * e_vec[n - 1]) / sub_diag[n - 2];
        for (size_t k = n - 2; k-- > 0;) {
            e_vec[k] = ((ev - main_diag[k + 1]) * e_vec[k + 1] - sub_diag[k + 1] * e_vec[k + 2]) / sub_diag[k];
        }
    }
};


struct RandomVectorGenerator
{
	curandGenerator_t gen;
	curandRngType rng = CURAND_RNG_PSEUDO_XORWOW;

	RandomVectorGenerator() {
        /* Create pseudo-random number generator */
        CURAND_CALL(curandCreateGenerator(&gen, rng));
        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	}

    void generate(float* arr, size_t n, float min, float max) const {
        CURAND_CALL(curandGenerateUniform(gen, arr, n));
        scale_vec(arr, max - min, min, n);
    }
};



class EigenSolver
{
public:
    /**
     * @brief Executes Lanczos algorithm and stores the result into reference variables passed as arguments.
     * @return Lanczos-iteration count
     */
    EigenSolver();
    size_t solve(float* arr, float* eigvalues, float* eigvecs, size_t mat_size, size_t num_ei, bool maximum) const;


private:
    RandomVectorGenerator rng;
    cublasHandle_t ctx = NULL;
    cudaStream_t stream = NULL;
};

