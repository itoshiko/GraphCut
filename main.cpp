#include "eigen/EigenSolver.h"
#include "kmeans/kmeans.hpp"
#include "graphLap.h"
#include "imageIO.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <io.h>
#include <direct.h>

int createDirectory(std::string path)
{
	int len = path.length();
	char tmpDirPath[256] = { 0 };
	for (int i = 0; i < len; i++)
	{
		tmpDirPath[i] = path[i];
		if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
		{
			if (_access(tmpDirPath, 0) == -1)
			{
				int ret = _mkdir(tmpDirPath);
				if (ret == -1) return ret;
			}
		}
	}
	return 0;
}

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

void make_param(std::vector<std::vector<GraphGLOption>>& opt_set, std::vector<std::string>& paths)
{
	// weight
	std::vector<float> weight{ 1.0, 0.8, 0.5, 0.2, 0.0 };
	opt_set.emplace_back();
	paths.emplace_back("D:/course_proj/GraphCut/test_img/mix_weight/");
	for (auto w : weight)
	{
		opt_set.back().emplace_back();
		opt_set.back().back().weight_int = w;
		opt_set.back().back().weight_dist = 1. - w;
	}

	// sigma
	std::vector<float> sigma{ 2.0, 1.0, 0.5, 0.4, 0.25 };
	opt_set.emplace_back();
	paths.emplace_back("D:/course_proj/GraphCut/test_img/sigma/");
	for (auto s : sigma)
	{
		opt_set.back().emplace_back();
		opt_set.back().back().weight = GLWeight::WEIGHT_INTENSITY;
		opt_set.back().back().sigma_int = s;
	}

	// cut off threshold
	std::vector<float> ths{ 0.99, 0.95, 0.8, 0.5, 0.0 };
	opt_set.emplace_back();
	paths.emplace_back("D:/course_proj/GraphCut/test_img/threshold/");
	for (auto t : ths)
	{
		opt_set.back().emplace_back();
		opt_set.back().back().weight = GLWeight::WEIGHT_INTENSITY;
		opt_set.back().back().weight_th = t;
	}

	// ncut or ratio cut
	opt_set.emplace_back();
	paths.emplace_back("D:/course_proj/GraphCut/test_img/cut_way/");
	opt_set.back().emplace_back();
	opt_set.back().back().weight = GLWeight::WEIGHT_INTENSITY;
	opt_set.back().back().type = GLType::RATIO_CUT;
	opt_set.back().emplace_back();
	opt_set.back().back().weight = GLWeight::WEIGHT_INTENSITY;
	opt_set.back().back().type = GLType::N_CUT;
}

void run_exp(std::vector<GraphGLOption> opts, std::string save_path)
{
	GraphLaplacian graphL;
	imageIO io;
	int height, width;
	std::vector<std::string> imgs{ "baseball_game.jpg", "person.jpg", "weather.jpg" };
	std::string path = "D:/course_proj/GraphCut/test_img/";
	createDirectory(save_path);
	printf("%s\n", save_path.c_str());

	size_t ei_num = 2;
	EigenSolver solver;

	for (auto image : imgs)
	{
		printf("  %s\n", image.c_str());
		nvjpegImage_t img = io.readImage(path + image, &width, &height);
		float* img_arr = nullptr;
		graphL.preprocessInput(img, &img_arr, height, width);
		int cnt = 0;
		for (auto opt : opts)
		{
			float* ev = nullptr;
			float* evec = nullptr;
			CUDA_CALL(cudaMalloc(&ev, ei_num * sizeof(float)));
			CUDA_CALL(cudaMalloc(&evec, ei_num * height * width * sizeof(float)));
			cnt++;
			float* lp = graphL.constructGL(img_arr, height, width, 1, opt);
			solver.solve(lp, ev, evec, height * width, ei_num, false);
			int* label = KMeans::bisec(evec + height * width, height * width, 0.);
			unsigned char* output = nullptr;
			io.visualizeResult(img_arr, label, &output, height, width, ei_num);
			io.writeImage(save_path + image.substr(0, image.size() - 4) + "_cut_" + std::to_string(cnt) + ".jpg", output, height, width);
		}
	}
}

void main_exp()
{
	std::vector<std::vector<GraphGLOption>> opt_set;
	std::vector<std::string> save_path;
	make_param(std::ref(opt_set), std::ref(save_path));
	for (int idx = 0; idx < opt_set.size(); ++idx)
	{
		run_exp(opt_set[idx], save_path[idx]);
	}
}

int seg_single(size_t ei_num)
{
	imageIO io;
	int height, width;
	nvjpegImage_t img = io.readImage("D:/course_proj/GraphCut/test_img/person.jpg", &width, &height);

	GraphLaplacian graphL;
	GraphGLOption opt;
	opt.type = GLType::N_CUT;
	opt.weight = GLWeight::WEIGHT_MIX;
	opt.mix = GLWeightMix::MIX_WEIGHTED_SUM;
	opt.weight_th = 0.8;
	opt.sigma_int = 0.5;
	opt.sigma_dist = 0.5;
	opt.weight_int = 0.7;
	opt.weight_dist = 0.3;
	float* img_arr = nullptr;
	graphL.preprocessInput(img, &img_arr, height, width);
	float* lp = graphL.constructGL(img_arr, height, width, 1, opt);
	
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

	int *label = KMeans::run(evec + height * width, ei_num - 1, ei_num - 1, height * width);
	// int* label = KMeans::bisec(evec + height * width, height * width, 0.);

	unsigned char* output = nullptr;
	io.visualizeResult(img_arr, label, &output, height, width, ei_num);
	io.writeImage(std::string("D:/course_proj/GraphCut/test_img/person_cut") + std::to_string(ei_num - 1) + ".jpg", output, height, width);
	return 0;
}

int main()
{
	for (int i = 3; i < 4; ++i)
		seg_single(i);
}