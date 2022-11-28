#pragma once
#include "kmeans_util.h"

class KMeans
{
public:
	static int* run(float* arr, size_t k_rep, size_t dim, size_t sample) {
		float* reps;
		int* label;
		float* distCache;
		CUDA_CALL(cudaMalloc(&reps, k_rep * dim * sizeof(float)));
		CUDA_CALL(cudaMalloc(&label, sample * sizeof(int)));
		initRep(arr, reps, k_rep, dim, sample);
		solveKMeans(arr, reps, label, k_rep, dim, sample, ITER_MAX);
		printf("KMeans solved\n");
		return label;
	}

private:
	static const size_t ITER_MAX = 100;
};