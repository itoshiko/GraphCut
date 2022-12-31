#pragma once
#include <random>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}

void ArrayMax(const float* input, float* max_val, size_t n);
void ArrayMin(const float* input, float* min_val, size_t n);
void ArrayArgmin(const float* input, float* min_val, int* min_idx, size_t n);
void ArrayArgmax(const float* input, float* max_val, int* max_idx, size_t n);

void initRep(
	float* arr, 
	float* reps, 
	size_t k_rep, 
	size_t dim, 
	size_t sample);

void initRepRandom(
	float* arr,
	float* reps,
	size_t k_rep,
	size_t dim,
	size_t sample);

void solveKMeans(
	float* arr,
	float* reps,
	int* label,
	size_t k_rep,
	size_t dim,
	size_t sample,
	size_t max_iter);

void transpose(
	float* in,
	float* out,
	size_t num_sample,
	size_t dim
);

void cutByTh(
	float* arr, 
	int* label, 
	size_t n, 
	float th);

