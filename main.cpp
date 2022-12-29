#include "eigen/EigenSolver.h"
#include "kmeans/kmeans.hpp"
#include "graphLap.h"
#include "imageIO.h"
#include <iostream>
#include <fstream>
#include <random>

void save_lp(float* lp, size_t height, size_t width)
{
	int mat_size = height * width;
	float* debug_lp = new float[mat_size * (mat_size + 1) / 2];
	cudaMemcpy(debug_lp, lp, (mat_size * (mat_size + 1) / 2) * sizeof(float), cudaMemcpyDeviceToHost);
	float* debug_lp_full = new float[mat_size * mat_size];
	for (int ii = 0; ii < mat_size; ii++)
	{
		for (int jj = 0; jj < mat_size; jj++)
		{
			if (ii == jj) debug_lp_full[ii * mat_size + jj] = debug_lp[ii * (ii + 3) / 2];
			else if (ii > jj) debug_lp_full[ii * mat_size + jj] = debug_lp[ii * (ii + 1) / 2 + jj];
			else debug_lp_full[ii * mat_size + jj] = debug_lp[jj * (jj + 1) / 2 + ii];
		}
	}
	FILE* fout = fopen("D:/course_proj/GraphCut/test_img/lp.bin", "wb");
	fwrite(debug_lp_full, sizeof(float), (mat_size * mat_size), fout);
	fclose(fout);
}

void save_eigen_vec(float* vec, size_t dim, size_t num)
{
	float* vec_host = new float[dim * num];
	cudaMemcpy(vec_host, vec, sizeof(float) * dim * num, cudaMemcpyDeviceToHost);
	FILE* fout = fopen("D:/course_proj/GraphCut/test_img/vec.bin", "wb");
	fwrite(vec_host, sizeof(float), dim * num, fout);
	fclose(fout);
}

int main()
{
	imageIO io;
	int height, width;
	nvjpegImage_t img = io.readImage("D:/course_proj/GraphCut/test_img/baseball_game.jpg", &width, &height);

	GraphLaplacian graphL;
	GraphGLOption opt;
	opt.type = GLType::N_CUT;
	opt.weight = GLWeight::WEIGHT_MIX;
	opt.mix = GLWeightMix::MIX_WEIGHTED_SUM;
	opt.weight_th = 0.8;
	opt.sigma_int = 0.3;
	opt.sigma_dist = 0.3;
	opt.weight_int = 0.7;
	opt.weight_dist = 0.3;
	float* img_arr = nullptr;
	graphL.preprocessInput(img, &img_arr, height, width);
	float* lp = graphL.constructGL(img_arr, height, width, 1, opt);

	size_t ei_num = 10;
	float* ev = nullptr;
	float* evec = nullptr;
	CUDA_CALL(cudaMalloc(&ev, ei_num * sizeof(float)));
	CUDA_CALL(cudaMalloc(&evec, ei_num * height * width * sizeof(float)));
	EigenSolver solver;
	solver.solve(lp, ev, evec, height * width, ei_num, false);
	float* ev_host = new float[ei_num];
	CUDA_CALL(cudaMemcpy(ev_host, ev, ei_num * sizeof(float), cudaMemcpyDeviceToHost));
	printf("Eigen solved.\n");
	for (int ii = 0; ii < ei_num; ++ii)
		printf("  Eigenvalue %d: %f\n", ii, ev_host[ii]);

	//save_lp(lp, height, width);
	//save_eigen_vec(evec, height * width, ei_num);
	//exit(0);
	
	//int *label = KMeans::run(evec + height * width, ei_num - 1, ei_num - 1, height * width);
	int* label = KMeans::bisec(evec + height * width, height * width, 0.);



	unsigned char* output = nullptr;
	io.visualizeResult(img_arr, label, &output, height, width, ei_num);
	io.writeImage("D:/course_proj/GraphCut/test_img/baseball_game_cut.jpg", output, height, width);
	return 0;
}