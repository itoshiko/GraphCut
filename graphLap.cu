#include "graphLap.h"

__global__
void _get_vec_one(float* arr, float sca, size_t n)
{
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (threadId < n) arr[threadId] = sca;
}

__global__
void _calc_aff_mat_rbf(float* arr, float* aff, float sigma, size_t n, size_t dim)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	if (p2 < p1 || p2 >= n) return;
	float sim = 0.;
	if (p1 == p2)
	{
		sim = 1.;
	}
	else
	{
		float* p1_data = arr + p1 * dim;
		float* p2_data = arr + p2 * dim;
		for (int d = 0; d < dim; d++)
			sim += (p1_data[d] - p2_data[d]) * (p1_data[d] - p2_data[d]);
		sim = expf(-sim / (2 * sigma * sigma));
	}
	aff[(p2 * (p2 + 1)) / 2 + p1] = sim;
}

__global__
void _calc_lp_mat(float* aff, float* deg, size_t n)
{
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (threadId < n) aff[(threadId * (threadId + 3)) / 2] += deg[threadId];
}

GraphLaplacian::GraphLaplacian()
{
	CUBLAS_CALL(cublasCreate(&ctx));
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CALL(cublasSetStream(ctx, stream));
	CUBLAS_CALL(cublasSetPointerMode(ctx, CUBLAS_POINTER_MODE_HOST));
}

void GraphLaplacian::constructGLGrad(float* arr, size_t height, size_t width, size_t dim)
{
	size_t mat_size = height * width;
	float* lp;
	CUDA_CALL(cudaMalloc(&lp, (mat_size * (mat_size + 1) / 2) * sizeof(float)));
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((mat_size - 1) / 32 + 1, (mat_size - 1) / 32 + 1);
	_calc_aff_mat_rbf << <gridSize, blockSize >> > (arr, lp, 1.0, mat_size, dim);
	float* degree;
	CUDA_CALL(cudaMalloc(&degree, mat_size * sizeof(float)));
	_get_vec_one << <dim3(32, 32, 1), (mat_size - 1) / (1024) + 1 >> > (degree, -1.0, mat_size);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	CUBLAS_CALL(cublasStpmv(ctx, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, mat_size, lp, degree, 1));
	_calc_lp_mat << <dim3(32, 32, 1), (mat_size - 1) / (1024) + 1 >> > (lp, degree, mat_size);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
}