#include "imageIO.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__
unsigned char f2uc(float f)
{
	f = f * 255.;
	if (f > 255.0) f = 255.;
	if (f < 0.) f = 0.;
	return (unsigned char)f;
}

__global__
void _make_output_img(float* img, int* label, unsigned char* result, float* color_map, size_t height, size_t width)
{
	int p1 = blockIdx.x * blockDim.x + threadIdx.x;
	int p2 = blockIdx.y * blockDim.y + threadIdx.y;
	if (p1 >= height || p2 >= width) return;
	int pix_idx = p1 * width + p2;
	float r = 0.5 * img[pix_idx] + 0.5 * color_map[label[pix_idx] * 3];
	float g = 0.5 * img[pix_idx] + 0.5 * color_map[label[pix_idx] * 3 + 1];
	float b = 0.5 * img[pix_idx] + 0.5 * color_map[label[pix_idx] * 3 + 2];
	result[pix_idx] = f2uc(r);
	result[pix_idx + width * height] = f2uc(g);
	result[pix_idx + width * height * 2] = f2uc(b);
}

imageIO::imageIO()
{
	CHECK_NVJPEG(nvjpegCreateSimple(&ctx));
	CHECK_NVJPEG(nvjpegJpegStateCreate(ctx, &jpegHandle));
	CUDA_CALL(cudaStreamCreate(&mStream));
	CHECK_NVJPEG(nvjpegEncoderStateCreate(ctx, &jpegEncodeHandle, mStream));
	initColorMap();
}

nvjpegImage_t imageIO::readImage(std::string path, int *width, int *height)
{
	nvjpegImage_t decoded;
	unsigned char* jpeg_data = nullptr;
	size_t jpeg_size = 0;
	readFile(path.c_str(), &jpeg_data, &jpeg_size);
	if (jpeg_data == nullptr || jpeg_size == 0)
	{
		printf("Failed reading file!\n");
		decoded.channel[0] = nullptr;
		return decoded;
	}

	uint32_t format;
	parse_jpeg_header(jpeg_data, jpeg_size, (uint32_t*)width, (uint32_t*)height, &format);
	printf("%d %d\n", *width, *height);

	void* decoded_buffer;
	CUDA_CALL(cudaMalloc(&decoded_buffer, *height * *width * sizeof(unsigned char)));
	decoded.channel[0] = (unsigned char*)decoded_buffer;
	decoded.pitch[0] = *width;
	CHECK_NVJPEG(nvjpegDecode(
		ctx, jpegHandle,
		jpeg_data, jpeg_size,
		NVJPEG_OUTPUT_Y, &decoded, mStream
	));
	CUDA_CALL(cudaStreamSynchronize(mStream));
	return decoded;
}

void imageIO::writeImage(std::string path, unsigned char* data, size_t height, size_t width)
{
	nvjpegImage_t jpegImg;
	for (int i = 0; i < 3; i++) {
		jpegImg.channel[i] = data + i * height * width;
		jpegImg.pitch[i] = width;
	}
	nvjpegEncoderParams_t jpegEncParam;
	nvjpegEncoderParamsCreate(ctx, &jpegEncParam, mStream);
	CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(jpegEncParam, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, mStream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(jpegEncParam, 1, mStream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(jpegEncParam, 95, mStream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(jpegEncParam, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, mStream));
	nvjpegStatus_t sta = nvjpegEncodeImage(ctx, jpegEncodeHandle, jpegEncParam, &jpegImg, NVJPEG_INPUT_RGB, width, height, mStream);

	// get compressed stream size
	size_t length;
	nvjpegEncodeRetrieveBitstream(ctx, jpegEncodeHandle, NULL, &length, mStream);

	// get stream itself
	CUDA_CALL(cudaStreamSynchronize(mStream));
	std::vector<unsigned char> jpeg(length);
	nvjpegEncodeRetrieveBitstream(ctx, jpegEncodeHandle, jpeg.data(), &length, mStream);

	// write stream to file
	CUDA_CALL(cudaStreamSynchronize(mStream));
	FILE* output_file = fopen(path.c_str(), "wb");
	fwrite(jpeg.data(), sizeof(unsigned char), length, output_file);
	fclose(output_file);
}

imageIO::~imageIO()
{
	CHECK_NVJPEG(nvjpegJpegStateDestroy(jpegHandle));
	CHECK_NVJPEG(nvjpegEncoderStateDestroy(jpegEncodeHandle));
	CHECK_NVJPEG(nvjpegDestroy(ctx));
}

void imageIO::visualizeResult(float* img, int* label, unsigned char** result, size_t height, size_t width, int num_patch)
{
	CUDA_CALL(cudaMalloc((void**)(result), height * width * sizeof(unsigned char) * 3));
	dim3 blockSize(32, 32, 1);
	dim3 gridSize((height - 1) / 32 + 1, (width - 1) / 32 + 1, 1);
	_make_output_img << <gridSize, blockSize >> > (img, label, *result, colorMap, height, width);
	CUDA_CALL(cudaDeviceSynchronize());
}

void imageIO::readFile(const char* path, unsigned char** result, size_t* size)
{
	FILE* pfile;
	pfile = fopen(path, "rb");
	if (pfile == NULL) {
		*result = nullptr;
		size = 0;
		return;
	}
	fseek(pfile, 0, SEEK_END);
	*size = ftell(pfile);
	*result = (unsigned char*)malloc((*size + 1) * sizeof(char));
	rewind(pfile);
	*size = fread(*result, 1, *size, pfile);
	(*result)[*size] = '\0';
	fclose(pfile);
}

void imageIO::initColorMap()
{
	float _c[]{
		176. / 255., 23. / 255., 31. / 255.,
		255. / 255., 153. / 255., 18. / 255.,
		34. / 255., 139. / 255., 34. / 255.,
		3. / 255., 168. / 255., 158. / 255.,
		255. / 255., 99. / 255., 71. / 255.,
		255. / 255., 215. / 255., 0.,
		0., 255. / 255., 255. / 255.,
		135. / 255., 38. / 255., 87. / 255.,
		218. / 255., 112. / 255., 214. / 255.,
		127. / 255., 255. / 255., 0.,
		250. / 255., 235. / 255., 215. / 255.
	};
	CUDA_CALL(cudaMalloc(&colorMap, sizeof(float) * 33));
	CUDA_CALL(cudaMemcpy(colorMap, _c, sizeof(float) * 33, cudaMemcpyHostToDevice));
}

void show_segment(uint32_t marker)
{
	int32_t index = marker - 0xc0;

	if (index < 0 || index >= sizeof(seg_name) / sizeof(char*))
		return;

	printf("%s\n", seg_name[index]);
}

void usage(void) {
	printf("usage: jpeg_parse file\n");

	return;
}

int parse_jpeg_header(uint8_t* data, uint32_t length, uint32_t* width, uint32_t* height, uint32_t* format)
{
	uint8_t* start, * end, * cur;
	int found = 0;

	start = data;
	end = data + length;
	cur = start;

	// printf("start - %p, cur - %p, end - %p\n", start, cur, end);

#define READ_U8(value) do \
	{ \
		(value) = *cur; \
		++cur; \
	} while (0)

#define READ_U16(value) do \
	{ \
		uint16_t w = *((uint16_t *)cur); \
		cur += 2; \
		(value) = (((w & 0xff) << 8) | ((w & 0xff00) >> 8)); \
		/* printf("w = 0x%x, value = 0x%x\n", w, value); */\
	} while (0)

	while (cur < end) {
		uint8_t marker;

		if (*cur++ != 0xff) {
			printf("%2x%2x->%2x%2x\n", *(cur - 2), *(cur - 1), *cur, *(cur + 1));;
			printf("cur pos: 0x%lx\n", cur - start);
			break;
		}

		READ_U8(marker);

		if (marker == JPEG_MARKER_SOS)
			break;

		show_segment(marker);

		switch (marker) {
		case JPEG_MARKER_SOI:
			break;
		case JPEG_MARKER_DRI:
			cur += 4;	/* |length[0..1]||rst_interval[2..3]|*/
			break;
		case JPEG_MARKER_SOF2:
			fprintf(stderr, "progressive JPEGs not suppoted\n");
			break;
		case JPEG_MARKER_SOF0: {
			uint16_t length;
			uint8_t sample_precision;

			READ_U16(length);
			length -= 2;

			READ_U8(sample_precision);
			printf("sample_precision = %d\n", sample_precision);

			READ_U16(*height);
			READ_U16(*width);
			length -= 5;

			cur += length;

			found = 1;
			break;
		}
		default: {
			uint16_t length;
			READ_U16(length);
			// printf("cur: 0x%lx, length: 0x%x\n", cur-start, length);
			length -= 2;
			cur += length;
			break;
		}
		}
	}

	printf("parse jpeg header finish\n");

	return found;
}