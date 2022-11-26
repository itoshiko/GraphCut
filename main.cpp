#include "EigenSolver.h"
#include <iostream>
#include <random>

int main()
{
	size_t mat_size = 10000;
	size_t ei_num = 10;
	std::random_device e;
	std::uniform_real_distribution<float> u(0, 1);
	size_t arr_size = (mat_size * (1 + mat_size)) / 2;
	float* arr_host = new float[arr_size];
	for (int i = 0; i < arr_size; i++)
		arr_host[i] = u(e);
	//float arr_host[] = { 5., 4., 3., 3., 5., 9., 6., 7., 6., 8. };
	float* arr = nullptr;
	cudaMalloc(&arr, arr_size * sizeof(float));
	cudaMemcpy(arr, arr_host, arr_size * sizeof(float), cudaMemcpyHostToDevice);
	float* ev = new float[ei_num];
	float* evec = new float[ei_num * mat_size];

	EigenSolver solver;
	clock_t t1 = clock();
	solver.solve(arr, ev, evec, mat_size, ei_num, false);
	printf("Elapsed %f s\n", (float)(clock() - t1) / (CLOCKS_PER_SEC));
	for (int i = 0; i < ei_num; i++) printf("Eigen %d: %f\n", i, ev[i]);
	return 0;
}