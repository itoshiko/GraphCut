#pragma once
#include "kmeans_util.h"
#include <fstream>

class KMeans
{
public:
	static int* run(float* arr, size_t k_rep, size_t dim, size_t sample) {
		float* reps;
		int* label;
		float* distCache;
		CUDA_CALL(cudaMalloc(&reps, k_rep * dim * sizeof(float)));
		CUDA_CALL(cudaMalloc(&label, sample * sizeof(int)));
		float* arr_trans;
		CUDA_CALL(cudaMalloc(&arr_trans, sample * dim * sizeof(float)));
		transpose(arr, arr_trans, sample, dim);
		initRep(arr_trans, reps, k_rep, dim, sample);
		solveKMeans(arr_trans, reps, label, k_rep, dim, sample, ITER_MAX);
		printf("KMeans solved\n");
		return label;
	}

	static int* bisec(float* arr, size_t sample, float th) {
		int* label;
		CUDA_CALL(cudaMalloc(&label, sample * sizeof(int)));
		cutByTh(arr, label, sample, th);
		return label;
	}

private:
	static const size_t ITER_MAX = 100;
};