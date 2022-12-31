#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <nvjpeg.h>
#include <iostream>

#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}

#define CUBLAS_CALL(x) if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n", x, __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}

enum class GLType { N_CUT, RATIO_CUT };
enum class GLWeight { WEIGHT_INTENSITY, WEIGHT_MIX };
enum class GLWeightMix { MIX_WEIGHTED_SUM, WEIGHT_MULTIPLY };

struct GraphGLOption 
{
    GLType type;
    GLWeight weight;
    GLWeightMix mix;
    float weight_int, weight_dist;
    float weight_th;
    float sigma_int, sigma_dist;

    GraphGLOption()
    {
        type = GLType::N_CUT;
        weight = GLWeight::WEIGHT_MIX;
        mix = GLWeightMix::MIX_WEIGHTED_SUM;
        weight_th = 0.6;
        sigma_int = 0.5;
        sigma_dist = 0.5;
        weight_int = 0.8;
        weight_dist = 0.2;
    }
};

class GraphLaplacian
{
public:
    GraphLaplacian();
    void preprocessInput(nvjpegImage_t img, float** arr, int height, int width);
    float* constructGL(float* arr, size_t height, size_t width, size_t dim, GraphGLOption opt);

private:
    cublasHandle_t ctx = NULL;
    cudaStream_t stream = NULL;
};
