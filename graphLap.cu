#include "graphLap.h"

__device__ __forceinline__
float _calc_pix_diff(float* p1, float* p2, float sigma, size_t dim)
{
	float sim = 0.;
	for (int d = 0; d < dim; d++)
		sim += (p1[d] - p2[d]) * (p1[d] - p2[d]);
	sim = expf(-sim / (2 * sigma * sigma));
	return sim;
}

__device__ __forceinline__
float _calc_dist_diff(int p1, int p2, int width, int height, float sigma)
{
	float x1 = float(p1 / width) / float(height);
	float y1 = float(p1 % width) / float(width);
	float x2 = float(p2 / width) / float(height);
	float y2 = float(p2 % width) / float(width);
	float dist = 0.;
	dist += (x1 - x2) * (x1 - x2);
	dist += (y1 - y2) * (y1 - y2);
	dist = expf(-dist / (2 * sigma * sigma));
	return dist;
}

__global__
void _get_vec_one(float* arr, float sca, size_t n)
{
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (threadId < n)
		arr[threadId] = sca;
}

__global__
void _vec_minus_05(float* arr, float* minus05, size_t n)
{
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (threadId < n)
	{
		if (fabsf(arr[threadId]) < 1e-6)
			minus05[threadId] = 999.;
		else
		{
			minus05[threadId] = 1. / (sqrtf(arr[threadId]));
		}
	}
}

__global__
void _calc_aff_mat_int_rbf(float* arr, float* aff, float sigma, size_t n, size_t dim, float th)
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
		sim = _calc_pix_diff(p1_data, p2_data, sigma, dim);
	}
	if (sim < th) sim = 0.;
	aff[(p2 * (p2 + 1)) / 2 + p1] = sim;
}

__global__
void _calc_aff_mat_rbf(
	float* arr, float* aff, 
	float sigma_int, float sigma_dist, 
	float weight_int, float weight_dist,
	int width, int height, size_t dim, float th)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	size_t n = height * width;
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
		float sim_int = _calc_pix_diff(p1_data, p2_data, sigma_int, dim);
		float sim_dist = _calc_dist_diff(p1, p2, width, height, sigma_dist);
		sim = (weight_int * sim_int + weight_dist * sim_dist) / (weight_dist + weight_dist);
	}
	if (sim < th) sim = 0.;
	aff[(p2 * (p2 + 1)) / 2 + p1] = sim;
}

__global__
void _calc_aff_mat_rbf_mul(
	float* arr, float* aff,
	float sigma_int, float sigma_dist,
	int width, int height, size_t dim, float th)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	size_t n = height * width;
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
		float sim_int = _calc_pix_diff(p1_data, p2_data, sigma_int, dim);
		float sim_dist = _calc_dist_diff(p1, p2, width, height, sigma_dist);
		sim = sim_dist * sim_int;
	}
	if (sim < th) sim = 0.;
	aff[(p2 * (p2 + 1)) / 2 + p1] = sim;
}

__global__
void _calc_lp_mat(float* aff, float* deg, size_t n)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	if (p2 < p1 || p2 >= n) return;
	if (p2 == p1) {
		aff[(p2 * (p2 + 1)) / 2 + p1] = (deg[p1] - aff[(p2 * (p2 + 1)) / 2 + p1]);
	}
	else aff[(p2 * (p2 + 1)) / 2 + p1] *= -1.;
}

__global__
void _calc_n_lp_mat(float* aff, float* deg, float* deg_minus05, size_t n)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	if (p2 < p1 || p2 >= n) return;
	if (p2 == p1)
		aff[(p2 * (p2 + 1)) / 2 + p1] = ((deg[p1] - aff[(p2 * (p2 + 1)) / 2 + p1]) * deg_minus05[p1] * deg_minus05[p1]);
	else
		aff[(p2 * (p2 + 1)) / 2 + p1] = (-aff[(p2 * (p2 + 1)) / 2 + p1] * deg_minus05[p1] * deg_minus05[p2]);
}

__global__
void _normalize_img(float* output, unsigned char* input, size_t height, size_t width)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	if (p1 >= height || p2 >= width) return;
	output[p1 * width + p2] = (float)input[p1 * width + p2] / 255.;
}

GraphLaplacian::GraphLaplacian()
{
	CUBLAS_CALL(cublasCreate(&ctx));
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CALL(cublasSetStream(ctx, stream));
	CUBLAS_CALL(cublasSetPointerMode(ctx, CUBLAS_POINTER_MODE_HOST));
}

void GraphLaplacian::preprocessInput(nvjpegImage_t img, float** arr, int height, int width)
{
	CUDA_CALL(cudaMalloc(arr, height * width * sizeof(float)));
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((height - 1) / 32 + 1, (width - 1) / 32 + 1, 1);
	_normalize_img << <gridSize, blockSize >> > (*arr, img.channel[0], height, width);
	CUDA_CALL(cudaDeviceSynchronize());
}

float* GraphLaplacian::constructGL(float* arr, size_t height, size_t width, size_t dim, GraphGLOption opt)
{
	size_t mat_size = height * width;
	float* lp;
	CUDA_CALL(cudaMalloc(&lp, (mat_size * (mat_size + 1) / 2) * sizeof(float)));
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((mat_size - 1) / 32 + 1, (mat_size - 1) / 32 + 1);

	if (opt.weight == GLWeight::WEIGHT_INTENSITY)
		_calc_aff_mat_int_rbf << <gridSize, blockSize >> > (
			arr, lp, 
			opt.sigma_int, 
			mat_size, dim, opt.weight_th);
	else if (opt.weight == GLWeight::WEIGHT_MIX)
	{
		if (opt.mix == GLWeightMix::MIX_WEIGHTED_SUM)
		{
			_calc_aff_mat_rbf << <gridSize, blockSize >> > (
				arr, lp,
				opt.sigma_int, opt.sigma_dist,
				opt.weight_int, opt.weight_dist,
				width, height, dim, opt.weight_th);
		}
		else if (opt.mix == GLWeightMix::WEIGHT_MULTIPLY)
		{
			_calc_aff_mat_rbf_mul << <gridSize, blockSize >> > (
				arr, lp,
				opt.sigma_int, opt.sigma_dist,
				width, height, dim, opt.weight_th);
		}
	}

	float* degree = nullptr;
	if (opt.type == GLType::RATIO_CUT) {
		CUDA_CALL(cudaMalloc(&degree, 2 * mat_size * sizeof(float)));
	}
	else if (opt.type == GLType::N_CUT) {
		CUDA_CALL(cudaMalloc(&degree, 3 * mat_size * sizeof(float)));
	}
	_get_vec_one << <(mat_size - 1) / (1024) + 1, dim3(32, 32, 1) >> > (degree, 1.0, mat_size);
	CUDA_CALL(cudaDeviceSynchronize());
	float _one = 1.;
	float _zero = 0.;
	CUBLAS_CALL(cublasSspmv(ctx, CUBLAS_FILL_MODE_UPPER, mat_size, &_one, lp, degree, 1, &_zero, degree + mat_size, 1));
	CUDA_CALL(cudaDeviceSynchronize());
	if (opt.type == GLType::N_CUT)
	{
		_vec_minus_05 << <(mat_size - 1) / (1024) + 1, dim3(32, 32, 1) >> > (degree + mat_size, degree + mat_size * 2, mat_size);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		_calc_n_lp_mat << <gridSize, blockSize >> > (lp, degree + mat_size, degree + mat_size * 2, mat_size);
	}
	else if (opt.type == GLType::RATIO_CUT)
	{
		_calc_lp_mat << <gridSize, blockSize >> > (lp, degree + mat_size, mat_size);
	}
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	return lp;
}
