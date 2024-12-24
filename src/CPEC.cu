#include <cuda.h> 
#include <iostream> 
#include <vector> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "deprecated/stb_image_resize.h"

#include "CPEC.h"

using namespace std;

void change(){

     string inputPath = "img1.png";
    int iw, ih, n;
    
    unsigned char *idata = stbi_load(inputPath.c_str(), &iw, &ih, &n, 0);

   
    
    int ow = iw / 2;
    int oh = ih / 2;
    auto *odata = (unsigned char *) malloc(ow * oh * n); 
    
    // 改变图片尺寸
    stbir_resize(idata, iw, ih, 0, odata, ow, oh, 0, STBIR_TYPE_UINT8, n, STBIR_ALPHA_CHANNEL_NONE, 0,
                 STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                 STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                 STBIR_COLORSPACE_SRGB, nullptr
    );

    string outputPath = "./output.png";
    // 写入图片
    stbi_write_png(outputPath.c_str(), ow, oh, n, odata, 0);

    stbi_image_free(idata);
    stbi_image_free(odata);
}

void sayMiku(){
  cout << "Miku!" << endl;
}

__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

void madd(){

  int c;
	int *gpu_c;
    // 使用cudaMalloc在gpu上分配内存
    // 第一个参数是一个指针，指向用于保存新分配内存地址的变量（传入一个指针的地址）；第二个参数是分配内存的大小
    // 注意，不可以在cpu上对gpu_c进行解引用！
    cudaMalloc((void **)&gpu_c, sizeof(int));
    add<<<1, 1>>>(39, 1, gpu_c);
    cudaMemcpy(&c, gpu_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Taller Miku: %d", c);
    cudaFree(gpu_c);
}