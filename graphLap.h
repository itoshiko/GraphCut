#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>

#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}

#define CUBLAS_CALL(x) if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n", x, __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}

class GraphLaplacian
{
public:
    GraphLaplacian();
	void constructGLGrad(float* arr, size_t height, size_t width, size_t dim);

private:
    cublasHandle_t ctx = NULL;
    cudaStream_t stream = NULL;
};
