#include "EigenSolver.h"


EigenSolver::EigenSolver()
{
    CUBLAS_CALL(cublasCreate(&ctx));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CALL(cublasSetStream(ctx, stream));
    CUBLAS_CALL(cublasSetPointerMode(ctx, CUBLAS_POINTER_MODE_HOST));
}

size_t EigenSolver::solve(float* arr, float* eig_values, float* eig_vecs, size_t mat_size, size_t num_ei, bool maximum) const
{
    size_t max_iter;
    if (mat_size < 1000)  max_iter = mat_size;
    else max_iter = mat_size;
    size_t vec_size = mat_size * sizeof(float);

    float* u; float* e_vec_tri_dev;
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

    //std::ofstream ofs;
    //ofs.open("D:/course_proj/GraphCut/test_img/iter_eval.txt", std::ios::out);
    //ofs << std::fixed << std::setprecision(8) << std::endl;

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
                alpha_host, beta_host, maximum ? max_iter - 1 - iroot : iroot, k, TOL);
            //ofs << evs[iroot] << " ";
        }
        //ofs << std::endl;

        if (beta_host[k - 1] < minimum_effective_decimal<float>() * 1e-2) {
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
            if (std::abs(ev - pev) >= std::min(std::abs(ev), std::abs(pev)) * 1e-6) {
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
    //ofs.close();
    beta_host[itern] = 0.0;
    e_vec_tri = new float[num_ei * itern];
    CUDA_CALL(cudaMalloc(&e_vec_tri_dev, num_ei * itern * sizeof(float)));

    for (size_t iroot = 0; iroot < num_ei; ++iroot) {
        TridiagonalEigen<float>::tridiagonal_eigenvector(alpha_host, beta_host, evs[iroot], e_vec_tri + (iroot * itern), itern);
    }
    CUDA_CALL(cudaMemcpy(e_vec_tri_dev, e_vec_tri, num_ei * itern * sizeof(float), cudaMemcpyHostToDevice));
    CUBLAS_CALL(cublasSgemm(
        ctx, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, num_ei, itern, &const_one, u, mat_size, e_vec_tri_dev, itern, &const_one, eig_vecs, mat_size));
    //for (size_t iroot = 0; iroot < num_ei; ++iroot) {
    //    normalize(eig_vecs + iroot * mat_size, mat_size, ctx);
    //}
    CUDA_CALL(cudaMemcpy(eig_values, evs, num_ei * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaDeviceSynchronize());

    delete[] evs;
    delete[] pevs;
    delete[] alpha_host;
    delete[] beta_host;
    delete[] e_vec_tri;
    CUDA_CALL(cudaFree(u));
    CUDA_CALL(cudaFree(e_vec_tri_dev));

    return itern;
}
