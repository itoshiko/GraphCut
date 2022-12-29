#pragma once
#include <nvjpeg.h>
#include <iostream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#define CHECK_NVJPEG(call)                                                  \
{                                                                           \
    nvjpegStatus_t _e = (call);                                             \
    if (_e != NVJPEG_STATUS_SUCCESS)                                        \
    {                                                                       \
        std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                            \
    }                                                                       \
}

#define CUDA_CALL(x) if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE);}

/* JPEG marker definitions. refer to itu-t81 Table B.1 */
/* Start Of Frame markers, non-differential, Huffman coding */
#define JPEG_MARKER_SOF0	0xc0	/* Baseline DCT */
#define JPEG_MARKER_SOF1	0xc1	/* Extended sequential DCT */
#define JPEG_MARKER_SOF2	0xc2	/* Progressive DCT */
#define JPEG_MARKER_SOF3	0xc3	/* Lossless(sequential) */

/* Start Of Frame markers, differential, Huffman coding */
#define JPEG_MARKER_SOF4	0xc5	/* Differential sequential DCT */
#define JPEG_MARKER_SOF5	0xc6	/* Differential progressive DCT */
#define JPEG_MARKER_SOF6	0xc7	/* Differential Lossless(sequential) */

/* Start Of Frame markers, non-differential, arithmetic coding */
#define JPEG_MARKER_JPG		0xc8	/* Reserved for JPEG extensions */
#define JPEG_MARKER_SOF9	0xc9	/* Extended sequential DCT */
#define JPEG_MARKER_SOF10	0xca	/* Progressive DCT */
#define JPEG_MARKER_SOF11	0xcb	/* Lossless(dequential) */

/* Start Of Frame markers, differential, arithmetic coding */
#define JPEG_MARKER_SOF13	0xcd	/* Differential sequential DCT */
#define JPEG_MARKER_SOF14	0xce	/* Differential progressive DCT */
#define JPEG_MARKER_SOF15	0xcf	/* Differential lossless(sequential) */

/* Restart interval termination */
#define JPEG_MARKER_RST0	0xd0	/* Restart ...*/
#define JPEG_MARKER_RST1	0xd1
#define JPEG_MARKER_RST2	0xd2
#define JPEG_MARKER_RST3	0xd3
#define JPEG_MARKER_RST4	0xd4
#define JPEG_MARKER_RST5	0xd5
#define JPEG_MARKER_RST6	0xd6
#define JPEG_MARKER_RST7	0xd7
#define JPEG_MARKER_RST8	0xd8
#define JPEG_MARKER_RST9	0xd9

/* Huffman table specification */
#define JPEG_MARKER_DHT		0xc4	/* Define Huffman table(s) */

/* Arithmetic coding conditioning specification */
#define JPEG_MARKER_DAC		0xcc	/* Define arithmetic coding conditioning(s) */

/* Other markers */
#define JPEG_MARKER_SOI		0xd8	/* Start of image */
#define JPEG_MARKER_EOI		0xd9	/* End of image */
#define JPEG_MARKER_SOS		0xda	/* Start of scan */
#define JPEG_MARKER_DQT		0xdb	/* Define quantization table(s) */
#define JPEG_MARKER_DNL		0xdc	/* Define number of lines */
#define JPEG_MARKER_DRI		0xdd	/* Define restart interval */
#define JPEG_MARKER_DHP		0xde	/* Define hierarchial progression */
#define JPEG_MARKER_EXP		0xdf	/* Expand reference component(s) */
#define JPEG_MARKER_APP0	0xe0	/* Application marker, JFIF/AVI1... */
#define JPEG_MARKER_APP1	0xe1	/* EXIF Metadata etc... */
#define JPEG_MARKER_APP2	0xe2	/* Not common... */
#define JPEG_MARKER_APP13	0xed	/* Photoshop Save As: IRB, 8BIM, IPTC */
#define JPEG_MARKER_APP14	0xee	/* Not common... */
#define JPEG_MARKER_APP15	0xef	/* Not common... */

static const char* seg_name[] = {
	"Baseline DCT; Huffman",
	"Extended sequential DCT; Huffman",
	"Progressive DCT; Huffman",
	"Spatial lossless; Huffman",
	"Huffman table",
	"Differential sequential DCT; Huffman",
	"Differential progressive DCT; Huffman",
	"Differential spatial; Huffman",
	"[Reserved: JPEG extension]",
	"Extended sequential DCT; Arithmetic",
	"Progressive DCT; Arithmetic",
	"Spatial lossless; Arithmetic",
	"Arithmetic coding conditioning",
	"Differential sequential DCT; Arithmetic",
	"Differential progressive DCT; Arithmetic",
	"Differential spatial; Arithmetic",
	"Restart",
	"Restart",
	"Restart",
	"Restart",
	"Restart",
	"Restart",
	"Restart",
	"Restart",
	"Start of image",
	"End of image",
	"Start of scan",
	"Quantisation table",
	"Number of lines",
	"Restart interval",
	"Hierarchical progression",
	"Expand reference components",
	"JFIF header",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: application extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"[Reserved: JPEG extension]",
	"Comment",
	"[Invalid]",
};

int parse_jpeg_header(uint8_t* data, uint32_t length, uint32_t* width, uint32_t* height, uint32_t* format);

class imageIO
{
public:
	imageIO();
    nvjpegImage_t readImage(std::string path, int* width, int* height);
    void writeImage(std::string path, unsigned char* data, size_t height, size_t width);
    ~imageIO();

    void visualizeResult(float* img, int* label, unsigned char** result, size_t height, size_t width, int num_patch);

private:
	nvjpegHandle_t ctx = nullptr;
	nvjpegJpegState_t jpegHandle = nullptr;
    nvjpegEncoderState_t jpegEncodeHandle = nullptr;
    cudaStream_t mStream = nullptr;
    void readFile(const char* path, unsigned char** result, size_t* size);
    
    float* colorMap = nullptr;
    void initColorMap();
};

