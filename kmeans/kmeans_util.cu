#include "kmeans_util.h"
#include <cub/cub.cuh>

template <class T> static __global__
void printDevice(T* _data, size_t cnt) {
	for (int idx = 0; idx < cnt; idx++)
		printf("%f ", float(*(_data + idx)));
	printf("\n");
}

__global__
void _update_dist_from_rep(float* arr, float* rep, float* dist, size_t dim, size_t sample, bool first)
{
	int pt = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (pt >= sample)
		return;
	float* start = arr + pt * dim;
	float _dist = 0.;
	for (int d = 0; d < dim; d++)
		_dist += fabsf(start[d] - rep[d]);
	if (first || dist[pt] > _dist) dist[pt] = _dist;
}

template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__
void _calc_centroid(float* arr, int* label, float* tmpSum, int* tmpCnt, float* new_center, size_t dim, size_t sample)
{
	int pt = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int d = blockIdx.y;
	int rid = blockIdx.z;
	if (pt < sample)
	{
		float _sum = 0.;
		int _cnt = 0;
		// Block wise reduction so that one thread in each block holds sum of thread results
		typedef cub::BlockReduce<float, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM_Y> BlockReduceF;
		typedef cub::BlockReduce<int, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM_Y> BlockReduceI;
		__shared__ typename BlockReduceF::TempStorage temp_storage_sum;
		__shared__ typename BlockReduceI::TempStorage temp_storage_cnt;

		// if (idx) belongs to patch (rep_id)
		if (label[pt] == rid)
		{
			_sum += arr[pt * dim + d];
			_cnt += 1;
		}
		float aggregate = BlockReduceF(temp_storage_sum).Sum(_sum);
		int total_cnt = BlockReduceI(temp_storage_cnt).Sum(_cnt);

		__syncthreads();
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			atomicAdd(tmpSum + rid * dim + d, aggregate);
			atomicAdd(tmpCnt + rid, total_cnt);
			__syncthreads();
			if (blockIdx.x == 0)
			{
				new_center[rid * dim + d] = (tmpSum[rid * dim + d] / (float)tmpCnt[rid]) * dim;
			}
		}
	}
}

__global__
void _assign_label(float* arr, float* rep, int* label, size_t k_rep, size_t dim, size_t sample)
{
	int pt = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (pt >= sample)
		return;
	float _minDist = -1.; int _label = -1;
	for (int i = 0; i < k_rep; i++)
	{
		float _dist = 0.;
		for (int j = 0; j < dim; j++)
			_dist += ((arr[dim * pt + j] - rep[dim * i + j]) * (arr[dim * pt + j] - rep[dim * i + j]));
		if (_dist < _minDist || _minDist < 0.)
		{
			_minDist = _dist;
			_label = i;
		}
	}
	label[pt] = _label;
}

__global__
void _transpose(float* in, float* out, size_t num_sample, size_t dim)
{
	int sp = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (sp >= num_sample) return;
	float sum = 0.;
	for (int i = 0; i < dim; ++i)
		sum += (in[num_sample * i + sp] * in[num_sample * i + sp]);
	sum = sqrtf(sum);
	for (int i = 0; i < dim; ++i)
		out[sp * dim + i] = in[num_sample * i + sp] / sum;
}

__global__
void _classify_th(float* arr, int* label, size_t n, float th)
{
	int sp = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (sp >= n) return;
	if (arr[sp] < th) label[sp] = 0;
	else label[sp] = 1;
}

void initRep(float* arr, float* reps, size_t k_rep, size_t dim, size_t sample)
{
	std::mt19937 rnd(time(nullptr));
	int32_t first = rnd() % sample;
	CUDA_CALL(cudaMemcpy(reps, arr + first * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice));
	if (k_rep == 1)
		return;
	float* distCache; float* tmpDist; int* tmpIdx;
	int rep_idx;
	CUDA_CALL(cudaMalloc(&distCache, sample * sizeof(float)));
	CUDA_CALL(cudaMalloc(&tmpDist, sizeof(float)));
	CUDA_CALL(cudaMalloc(&tmpIdx, sizeof(int)));
	dim3 blockSize(32, 16, 1);
	int gridSize = (sample - 1) / (32 * 16) + 1;
	for (int i = 1; i < k_rep; i++)
	{
		_update_dist_from_rep << <gridSize, blockSize >> > (arr, reps + (i - 1) * dim, distCache, dim, sample, i == 1 ? true : false);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		ArrayArgmax(distCache, tmpDist, tmpIdx, sample);
		CUDA_CALL(cudaMemcpy(&rep_idx, tmpIdx, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(reps + dim * i, arr + rep_idx * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
}

void solveKMeans(float* arr, float* reps, int* label, size_t k_rep, size_t dim, size_t sample, size_t max_iter)
{
	dim3 blockSize(32, 16, 1);
	int gridSize = (sample - 1) / (32 * 16) + 1;
	dim3 blockAvg(16, 16, 1);
	dim3 gridAvg((sample - 1) / (16 * 16) + 1, dim, k_rep);
	_assign_label << <gridSize, blockSize >> > (arr, reps, label, k_rep, dim, sample);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	float* tmpSum;
	int* tmpCnt;
	CUDA_CALL(cudaMalloc(&tmpSum, dim * k_rep * sizeof(float)));
	CUDA_CALL(cudaMalloc(&tmpCnt, k_rep * sizeof(int)));
	float* oldCenter = new float[dim * k_rep];
	float* centerHost = new float[dim * k_rep];
	for (int i = 0; i < max_iter; i++)
	{
		CUDA_CALL(cudaMemcpy(oldCenter, reps, dim * k_rep * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemset(tmpSum, 0, dim * k_rep * sizeof(float)));
		CUDA_CALL(cudaMemset(tmpCnt, 0, k_rep * sizeof(int)));
		_calc_centroid<16, 16> << <gridAvg, blockAvg >> > (arr, label, tmpSum, tmpCnt, reps, dim, sample);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		_assign_label << <gridSize, blockSize >> > (arr, reps, label, k_rep, dim, sample);
		CUDA_CALL(cudaMemcpy(centerHost, reps, dim * k_rep * sizeof(float), cudaMemcpyDeviceToHost));
		bool converge = true;
		for (int i = 0; i < k_rep; i++)
		{
			for (int j = 0; j < dim; j++)
				printf("%f ", centerHost[i * dim + j]);
			printf("\n");
		}
		for (int r = 0; r < k_rep; r++)
		{
			float _offset = 0.;
			for (int d = 0; d < dim; d++)
			{
				_offset += powf((oldCenter[r * dim + d] - centerHost[r * dim + d]), 2.);
			}
			_offset = sqrtf(_offset);
			if (_offset > 1e-5)
			{
				converge = false;
				printf("ITER %d disp %f\n", i, _offset);
				break;
			}
		}
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		if (converge) break;
	}
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
}

void ArrayMax(const float* input, float* max_val, size_t n) {
	size_t temp_storage_bytes;
	float* temp_storage = NULL;
	cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, input, max_val, n);
	CUDA_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
	cudaDeviceSynchronize();
	cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, input, max_val, n);
	cudaDeviceSynchronize();
}

void ArrayMin(const float* input, float* min_val, size_t n) {
	size_t temp_storage_bytes;
	float* temp_storage = NULL;
	cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, input, min_val, n);
	CUDA_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
	cudaDeviceSynchronize();
	cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, input, min_val, n);
	cudaDeviceSynchronize();
}

void ArrayArgmin(const float* input, float* min_val, int* min_idx, size_t n) {
	int BLOCK_SIZE = 32;
	int _grid = int(sqrt(double((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
	dim3 gridSize(_grid, _grid, 1);
	dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

	size_t temp_storage_bytes;
	float* temp_storage = NULL;
	cub::KeyValuePair<int, float>* h_out = new cub::KeyValuePair<int, float>;
	cub::KeyValuePair<int, float>* d_out;
	cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>));
	cub::DeviceReduce::ArgMin(temp_storage, temp_storage_bytes, input, d_out, n);
	CUDA_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
	cudaDeviceSynchronize();
	cub::DeviceReduce::ArgMin(temp_storage, temp_storage_bytes, input, d_out, n);
	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(min_idx, &(d_out->key), sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(min_val, &(d_out->value), sizeof(float), cudaMemcpyDeviceToHost));
}

void ArrayArgmax(const float* input, float* max_val, int* max_idx, size_t n) {
	int BLOCK_SIZE = 32;
	int _grid = int(sqrt(double((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1))) + 1;
	dim3 gridSize(_grid, _grid, 1);
	dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

	size_t temp_storage_bytes;
	float* temp_storage = NULL;
	cub::KeyValuePair<int, float>* h_out = new cub::KeyValuePair<int, float>;
	cub::KeyValuePair<int, float>* d_out;
	cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>));
	cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, input, d_out, n);
	CUDA_CALL(cudaMalloc(&temp_storage, temp_storage_bytes));
	cudaDeviceSynchronize();
	cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, input, d_out, n);
	cudaDeviceSynchronize();

	CUDA_CALL(cudaMemcpy(max_idx, &(d_out->key), sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(max_val, &(d_out->value), sizeof(float), cudaMemcpyDeviceToHost));
}

void transpose(float* in, float* out, size_t num_sample, size_t dim)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((num_sample - 1) / (32 * 32) + 1, 1, 1);
	_transpose << <gridSize, blockSize >> > (in, out, num_sample, dim);
	CUDA_CALL(cudaDeviceSynchronize());
}

void cutByTh(float* arr, int* label, size_t n, float th)
{
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((n - 1) / (32 * 32) + 1, 1, 1);
	_classify_th << <gridSize, blockSize >> > (arr, label, n, th);
	CUDA_CALL(cudaDeviceSynchronize());
}