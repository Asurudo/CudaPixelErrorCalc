#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>

#include "CPEC.h"

#define TEST_MODE true

using namespace std;

#if !TEST_MODE
int main(int argc, char** argv){
    if( argc < 5 ) {
		std::cout << "Usage: ./cpec.exe [reference_filename] [target_filename] [diffval|diffimg] [method] [cpu|gpu]" << std::endl;
		exit( - 1 );
	}

    std::string ref_filename = argv[ 1 ];
	std::string tgt_filename = argv[ 2 ];
	std::string mode = argv[ 3 ];
    std::string method = argv[ 4 ];
    std::string device = argv[ 5 ];
    std::string diffImgStyle  = ( argc < 7 ) ? "" : argv[ 6 ] ;
	float diffImgMin = ( argc < 8 ) ? -1.f : std::stof( argv[ 7 ] );
	float diffImgMax = ( argc < 9 ) ? -1.f : std::stof( argv[ 8 ] );

    std::unique_ptr<cpec::cpecImg> imgPtr1 = cpec::loadImage(ref_filename);
    std::unique_ptr<cpec::cpecImg> imgPtr2 = cpec::loadImage(tgt_filename);

    if(mode=="diffval"){
        double imgError = cpec::diffValCal(imgPtr1, imgPtr2, method, device=="cpu"?1:0);
        cout << method << " Error: " << imgError << endl;
    }
    else if(mode=="diffimg"){
        cpec::diffImgOutput(imgPtr1, imgPtr2, method, "./diffImgOutput.png", device=="cpu"?1:0,
        diffImgStyle==""?"PARULA":diffImgStyle, diffImgMin<0.f?0.f:diffImgMin, diffImgMax<0.f?0.f:diffImgMax);
    }
    else{
        std::cout << "Usage: ./cpec.exe [reference_filename] [target_filename] [diffval|diffimg] [method] [cpu|gpu]" << std::endl;
        exit( - 1 );
    }
    return 0;
}
#endif

#if TEST_MODE
int main() {
    string input1Path = "./big.jpg";
    string input2Path = "./big2.jpg";
    string method = "SSIM";
    //string outputPath = "./1.jpg";

    std::unique_ptr<cpec::cpecImg> imgPtr1 = cpec::loadImage(input1Path);
    std::unique_ptr<cpec::cpecImg> imgPtr2 = cpec::loadImage(input2Path);

    cout << "Image: " << imgPtr1->width << " " << imgPtr1->height << endl;
    
    //cpec::saveImage(imgPtr, outputPath);

    // 获取开始时间
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    double cpu_error = cpec::diffValCal(imgPtr1, imgPtr2, method, 1, 8);

    // 获取结束时间
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // 获取开始时间
    auto gpu_start = std::chrono::high_resolution_clock::now();

    double gpu_error = cpec::diffValCal(imgPtr1, imgPtr2, method, 0, 8);

    // 获取结束时间
    auto gpu_end = std::chrono::high_resolution_clock::now();

    // 计算时间差并输出
    std::chrono::duration<double> elapsed_cpu = cpu_end - cpu_start;
    std::cout << "CPU Elapsed time: " << elapsed_cpu.count() << " seconds" << std::endl;
    printf("%s CPU error: %.8f\n", method.c_str(), cpu_error);

    // 计算时间差并输出
    std::chrono::duration<double> elapsed_gpu = gpu_end - gpu_start;
    std::cout << "GPU Elapsed time: " << elapsed_gpu.count() << " seconds" << std::endl;
    printf("%s GPU error: %.8f\n", method.c_str(), gpu_error);
    //cpec::diffImgOutput(imgPtr1, imgPtr2, "./output.png");
    //cout << "MSE: " << error << endl;
    system("pause");
    return 0;
}
#endif

