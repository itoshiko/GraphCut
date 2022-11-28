#include "EigenSolver.h"


EigenSolver::EigenSolver()
{
    CUBLAS_CALL(cublasCreate(&ctx));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CALL(cublasSetStream(ctx, stream));
    CUBLAS_CALL(cublasSetPointerMode(ctx, CUBLAS_POINTER_MODE_HOST));
}

size_t EigenSolver::solve(float* arr, float* eigvalues, float* eigvecs, size_t mat_size, size_t num_ei, bool maximum) const
{
    size_t max_iter;
    if (mat_size < 1000)  max_iter = mat_size;
    else max_iter = mat_size;
    size_t vec_size = mat_size * sizeof(float);

    float* u; float* e_vec_tri_dev; float* e_vec_dev;
    float* alpha_host; float* beta_host; float* evs; float* pevs; float* e_vec_tri;
    CUDA_CALL(cudaMalloc(&u, (max_iter + 1) * vec_size));  // Lanczos vectors
    evs = new float[num_ei] {0.0};  // Calculated eigenvalue
    pevs = new float[num_ei] {0.0};  // Previous eigenvalue
    alpha_host = new float[max_iter] {0.0};  // Diagonal elements of an approximated tridiagonal matrix
    beta_host = new float[max_iter] {0.0};  // Subdiagonal elements of an approximated tridiagonal matrix

    CUDA_CALL(cudaMemset(u, 0, max_iter * vec_size));
    rng.generate(u, mat_size, -1.0, 1.0);
    normalize(u, mat_size, ctx);

    size_t itern = max_iter;
    float const_one = 1.0;
    for (size_t k = 1; k <= max_iter; ++k) {
        float* u_km1 = u + (k - 1) * mat_size;
        float* u_k = u + k * mat_size;
        /* au = (A + offset*E)uk, here E is the identity matrix */
        //print_vec_f_dev(u, mat_size);
        CUBLAS_CALL(cublasSspmv(ctx, CUBLAS_FILL_MODE_UPPER, mat_size, &const_one, arr, u_km1, 1, &const_one, u_k, 1));
        //print_vec_f_dev(u_k, mat_size);
        CUBLAS_CALL(cublasSdot(ctx, mat_size, u_km1, 1, u_k, 1, alpha_host + (k - 1)));
        // printf("alpha %f\n", alpha_host[k - 1]);
        if (k == 1) {
            float _alpha = -1. * alpha_host[k - 1];
            CUBLAS_CALL(cublasSaxpy(ctx, mat_size, &_alpha, u_km1, 1, u_k, 1));
        }
        else {
            float _alpha = -1. * alpha_host[k - 1];
            float _beta = -1. * beta_host[k - 2];
            CUBLAS_CALL(cublasSaxpy(ctx, mat_size, &_alpha, u_km1, 1, u_k, 1));
            CUBLAS_CALL(cublasSaxpy(ctx, mat_size, &_beta, u + (k - 2) * mat_size, 1, u_k, 1));
        }
        //print_vec_f_dev(u_k, 3);
        schmidt_orth(u, u_k, mat_size, k, ctx);
        //print_vec_f_dev(u, 3 * (k + 1));
        CUBLAS_CALL(cublasSnrm2(ctx, mat_size, u_k, 1, beta_host + (k - 1)));
        // printf("beta %f\n", beta_host[k - 1]);

        // find eigenvalue of tridiagonal matrix
        for (size_t iroot = 0; iroot < num_ei; ++iroot) {
            evs[iroot] = TridiagonalEigen<float>::find_mth_eigenvalue(
                alpha_host, beta_host, maximum ? max_iter - 1 - iroot : iroot, k, EPS);
            //std::cout << "ev " << iroot << " " << evs[iroot] << std::endl;
        }

        if (beta_host[k - 1] < minimum_effective_decimal<float>() * 1e-1) {
            itern = k;
            break;
        }
        normalize(u_k, mat_size, ctx);

        /*
         * only break loop if convergence condition is met for all roots
         */
        bool break_cond = true;
        for (size_t iroot = 0; iroot < num_ei; ++iroot) {
            const auto& ev = evs[iroot];
            const auto& pev = pevs[iroot];
            if (std::abs(ev - pev) >= std::min(std::abs(ev), std::abs(pev)) * 1e-4) {
                break_cond = false;
                break;
            }
        }

        if (break_cond) {
            itern = k;
            break;
        }
        else {
            memcpy_s(pevs, sizeof(float) * num_ei, evs, sizeof(float) * num_ei);
        }
    }

    memcpy_s(eigvalues, sizeof(float) * num_ei, evs, sizeof(float) * num_ei);
    beta_host[itern] = 0.0;
    e_vec_tri = new float[num_ei * itern];
    CUDA_CALL(cudaMalloc(&e_vec_tri_dev, num_ei * itern * sizeof(float)));
    CUDA_CALL(cudaMalloc(&e_vec_dev, num_ei * mat_size * sizeof(float)));
    CUDA_CALL(cudaMemset(e_vec_dev, 0, num_ei * mat_size * sizeof(float)));

    for (size_t iroot = 0; iroot < num_ei; ++iroot) {
        TridiagonalEigen<float>::tridiagonal_eigenvector(alpha_host, beta_host, eigvalues[iroot], e_vec_tri + (iroot * itern), itern);
    }
    CUDA_CALL(cudaMemcpy(e_vec_tri_dev, e_vec_tri, num_ei * itern * sizeof(float), cudaMemcpyHostToDevice));
    CUBLAS_CALL(cublasSgemm(
        ctx, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, num_ei, itern, &const_one, u, mat_size, e_vec_tri_dev, itern, &const_one, e_vec_dev, mat_size));
    for (size_t iroot = 0; iroot < num_ei; ++iroot) {
        normalize(e_vec_dev + iroot * mat_size, mat_size, ctx);
    }
    CUDA_CALL(cudaMemcpy(eigvecs, e_vec_dev, num_ei * mat_size * sizeof(float), cudaMemcpyDeviceToHost));
    return itern;
}
