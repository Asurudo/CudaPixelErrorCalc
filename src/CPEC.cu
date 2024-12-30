#include <cuda.h> 
#include <iostream> 
#include <vector> 
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "deprecated/stb_image_resize.h"

#include "CPEC.h"
#include "pseudocolor.hpp"

namespace cpecInter{
    #if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    #else
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                  __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
    #endif


    __global__ void gpu_diffValCalMSE_kernel(
      const cpec::col4 *img1_data, const cpec::col4 *img2_data,
      int width, int height, double *mse_result) {

      // 获取线程的 x 和 y 坐标
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      // 计算像素的全局下标
      int offset = x + y * width;

      // 如果线程在有效范围内
      if (x < width && y < height) {
          const cpec::col4 &p1 = img1_data[offset];
          const cpec::col4 &p2 = img2_data[offset];

          // 计算每个通道的平方误差
          double mse = (pow(p1.r - p2.r, 2) + pow(p1.g - p2.g, 2) + pow(p1.b - p2.b, 2));

          // 使用原子操作将每个线程计算的 MSE 累加到结果中
          atomicAdd(mse_result, mse);
      }
    }

    double gpu_diffValCalMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      if (img1->width != img2->width || img1->height != img2->height) {
          std::cerr << "Image dimensions do not match!" << std::endl;
          return -1.0f;
      }

      // 在设备上分配内存
      cpec::col4 *d_img1, *d_img2;
      double *d_mse_result;
      cudaMalloc((void**)&d_img1, img1->width * img1->height * sizeof(cpec::col4));
      cudaMalloc((void**)&d_img2, img1->width * img1->height * sizeof(cpec::col4));
      cudaMalloc((void**)&d_mse_result, sizeof(double));

      // 初始化 MSE 结果
      double MSE = 0.0f;
      cudaMemcpy(d_mse_result, &MSE, sizeof(double), cudaMemcpyHostToDevice);

      // 将图像数据从主机传输到设备
      cudaMemcpy(d_img1, img1->data.get(), img1->width * img1->height * sizeof(cpec::col4), cudaMemcpyHostToDevice);
      cudaMemcpy(d_img2, img2->data.get(), img1->width * img1->height * sizeof(cpec::col4), cudaMemcpyHostToDevice);

      int block_size = 32;
      // 计算所需的线程块和线程数量
      dim3 blocks((img1->width + block_size-1) / block_size, (img1->height + block_size-1) / block_size);
      dim3 threads(block_size, block_size);

      // 调用 GPU 核函数计算 MSE
      gpu_diffValCalMSE_kernel<<<blocks, threads>>>(d_img1, d_img2, img1->width, img1->height, d_mse_result);

      // 将 MSE 从设备复制到主机
      cudaMemcpy(&MSE, d_mse_result, sizeof(double), cudaMemcpyDeviceToHost);

      // 释放设备内存
      cudaFree(d_img1);
      cudaFree(d_img2);
      cudaFree(d_mse_result);

      return MSE / (3.0 * img1->width * img1->height);  // 除以总像素数得到最终的 MSE
    }
    double cpu_diffValCalMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      if (img1->width != img2->width || img1->height != img2->height) {
          std::cerr << "Image dimensions do not match!" << std::endl;
          return -1.0f;
      }

      double MSE = 0.0f;
      for (int y = 0; y < img1->height; ++y) {
          for (int x = 0; x < img1->width; ++x) {
              const cpec::col4& p1 = (*img1)(x, y);
              const cpec::col4& p2 = (*img2)(x, y);
              MSE += (std::pow(p1.r - p2.r, 2) + std::pow(p1.g - p2.g, 2) + std::pow(p1.b - p2.b, 2));
          }
      }

      MSE /= (3.0 * img1->width * img1->height);
      //std::cout << "MSE: " << MSE << std::endl;
      return MSE;
    }

    double gpu_diffValCalPSNR(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      double MSE = gpu_diffValCalMSE(img1, img2);
      if (MSE < 0) return -1.0f;
      double max_value = 255.0f;
      double psnr = 10 * std::log10((max_value * max_value) / MSE);
      //std::cout << "PSNR: " << psnr << std::endl;
      return psnr;
    }
    double cpu_diffValCalPSNR(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      double MSE = cpu_diffValCalMSE(img1, img2);
      if (MSE < 0) return -1.0f;
      double max_value = 255.0f;
      double psnr = 10 * std::log10((max_value * max_value) / MSE);
      //std::cout << "PSNR: " << psnr << std::endl;
      return psnr;
    }

    double gpu_diffValCalRMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      double MSE = gpu_diffValCalMSE(img1, img2);
      if (MSE < 0) return -1.0f;
      double RMSE = std::sqrt(MSE);
      //std::cout << "RMSE: " << RMSE << std::endl;
      return RMSE;
    }
    double cpu_diffValCalRMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2){
      double MSE = cpu_diffValCalMSE(img1, img2);
      if (MSE < 0) return -1.0f;
      double RMSE = std::sqrt(MSE);
      //std::cout << "RMSE: " << RMSE << std::endl;
      return RMSE;
    }

    __global__ void gpu_diffValCalSSIM_kernel(const cpec::col4* img1, const cpec::col4* img2, 
    int width, int height, int block_size, double c1, double c2, double* block_ssim) {
        int block_id_x = blockIdx.x * blockDim.x + threadIdx.x;
        int block_id_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (block_id_x >= (double)width / (double)block_size || block_id_y >= (double)height / (double)block_size) return;

        int block_start_x = block_id_x * block_size;
        int block_start_y = block_id_y * block_size;

        double mu0 = 0.0, mu1 = 0.0, sigma0 = 0.0, sigma1 = 0.0, sigma01 = 0.0;
        double inv = 1.0 / (block_size * block_size);
        
        for (int y = 0; y < block_size; ++y) {
            for (int x = 0; x < block_size; ++x) {
                int idx = (block_start_y + y) * width + (block_start_x + x);
                if(block_start_y + y >= height || block_start_x + x >= width) continue;
                double p1 = img1[idx].r;
                double p2 = img2[idx].r;

                mu0 += p1;
                mu1 += p2;
                sigma0 += p1 * p1;
                sigma1 += p2 * p2;
                sigma01 += p1 * p2;
            }
        }

        mu0 *= inv;
        mu1 *= inv;
        sigma0 = sigma0 * inv - mu0 * mu0;
        sigma1 = sigma1 * inv - mu1 * mu1;
        sigma01 = sigma01 * inv - mu0 * mu1;
      
        double ssim = (2.0 * mu0 * mu1 + c1) * (2.0 * sigma01 + c2) /
                      ((mu0 * mu0 + mu1 * mu1 + c1) * (sigma0 + sigma1 + c2));

        int block_idx = block_id_y * (width/ block_size) + block_id_x;
        block_ssim[block_idx] = ssim;
    }
    double gpu_diffValCalSSIM(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, int block_size = 8) {
        if (img1->width != img2->width || img1->height != img2->height) {
            std::cerr << "Image dimensions do not match!" << std::endl;
            return -1.0;
        }

        int width = img1->width;
        int height = img1->height;
        int img_size = width * height * sizeof(cpec::col4);
        int block_count = ((width + block_size - 1)/ block_size) * ((height + block_size - 1)/ block_size);
        int block_ssim_size = block_count * sizeof(double);

        double max_value = 255.0;
        double c1 = (0.01 * max_value) * (0.01 * max_value);
        double c2 = (0.03 * max_value) * (0.03 * max_value);

        cpec::col4 *d_img1, *d_img2;
        double* d_block_ssim;

        cudaMalloc((void**)&d_img1, img_size);
        cudaMalloc((void**)&d_img2, img_size);
        cudaMalloc((void**)&d_block_ssim, block_ssim_size);

        cudaMemcpy(d_img1, img1->data.get(), img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img2, img2->data.get(), img_size, cudaMemcpyHostToDevice);
      
        dim3 block_dim(32, 32);
        dim3 grid_dim(((double)width / (double)block_size + block_dim.x - 1) / block_dim.x,
                      ((double)height / (double)block_size + block_dim.y - 1) / block_dim.y);

        gpu_diffValCalSSIM_kernel<<<grid_dim, block_dim>>>(d_img1, d_img2, width, height, block_size, c1, c2, d_block_ssim);

        std::unique_ptr<double[]> h_block_ssim(new double[block_count]);
        cudaMemcpy(h_block_ssim.get(), d_block_ssim, block_ssim_size, cudaMemcpyDeviceToHost);

        double ssim_sum = 0.0;
        for (int i = 0; i < block_count; ++i) {
            ssim_sum += h_block_ssim[i];
        }
        ssim_sum /= block_count;

        cudaFree(d_img1);
        cudaFree(d_img2);
        cudaFree(d_block_ssim);

        return ssim_sum;
    }
    double cpu_diffValCalSSIM(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, int block_size = 8){
      if (img1->width != img2->width || img1->height != img2->height) {
          std::cerr << "Image dimensions do not match!" << std::endl;
          return -1.0f;
      }

      const double inv = 1.0f / (block_size * block_size);
      double ssim_sum = 0.0f;
      int id = 0;
      double max_value = 255.0f; // Assuming the max pixel value is 255
      double c1 = (0.01f * max_value) * (0.01f * max_value);
      double c2 = (0.03f * max_value) * (0.03f * max_value);

      for (int h = 0; h < img1->height - block_size; ++h) {
          for (int w = 0; w < img1->width - block_size; ++w) {
              double mu0 = 0.0f, mu1 = 0.0f, sigma0 = 0.0f, sigma1 = 0.0f, sigma01 = 0.0f;

              // Compute the mean and variance of the blocks
              for (int y = h; y < h + block_size; ++y) {
                  for (int x = w; x < w + block_size; ++x) {
                      const cpec::col4& p1 = (*img1)(x, y);
                      const cpec::col4& p2 = (*img2)(x, y);
                      mu0 += p1.r;
                      mu1 += p2.r;
                      sigma0 += p1.r * p1.r;
                      sigma1 += p2.r * p2.r;
                      sigma01 += p1.r * p2.r;
                  }
              }

              mu0 *= inv;
              mu1 *= inv;
              sigma0 = sigma0 * inv - mu0 * mu0;
              sigma1 = sigma1 * inv - mu1 * mu1;
              sigma01 = sigma01 * inv - mu0 * mu1;

              // Calculate SSIM for the block
              double ssim = (2.0f * mu0 * mu1 + c1) * (2.0f * sigma01 + c2) /
                          ((mu0 * mu0 + mu1 * mu1 + c1) * (sigma0 + sigma1 + c2));
              ssim_sum += ssim;
              ++id;
          }
      }

      ssim_sum /= id;
      //std::cout << "SSIM: " << ssim_sum << std::endl;
      return ssim_sum;
  }

    // 一律看成四通道图像进行比较，低通道的情况下默认使用第一个通道填充后面的通道
    std::unique_ptr<cpec::cpecImg> cpu_diffImgOutputMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap){
        // 创建一个新的cpecImg来存储差异结果
        std::unique_ptr<cpec::cpecImg> diffImg = std::make_unique<cpec::cpecImg>(img1->width, img1->height, 4, std::make_unique<cpec::col4[]>(img1->width * img1->height));

      // 计算每个像素的差异并应用颜色变换
      //float mse = 0.0f;
      const float epsilon = 1e-3f;
      for (int y = 0; y < img1->height; ++y) {
          for (int x = 0; x < img1->width; ++x) {
              cpec::col4& pixel1 = (*img1)(x, y);
              cpec::col4& pixel2 = (*img2)(x, y);
              cpec::col4& pixelDiff = (*diffImg)(x, y);

              const float Lr = 0.2126f * pixel1.r + 0.7152f * pixel1.g + 0.0722f * pixel1.b;
              const float Lt = 0.2126f * pixel2.r + 0.7152f * pixel2.g + 0.0722f * pixel2.b;
              if( !( std::isfinite( Lr * Lr ) ) ||  !( std::isfinite( Lt * Lt ))) continue;

              //mse += ( Lr - Lt ) * ( Lr - Lt ); //if( std::isnan( mse ) || !std::isfinite( mse ) ) { std::cout << mse << ", " << Lr << ", " << Lt << "\n"; break; }

              const auto col = colormap( ( Lr - Lt ) * ( Lr - Lt ));

              pixelDiff.r = col.x;
              pixelDiff.g = col.y; 
              pixelDiff.b = col.z;

              pixelDiff.a = 1.0f;  // 保持透明度为1.0
          }
      }
      return diffImg;
    }
    std::unique_ptr<cpec::cpecImg> gpu_diffImgOutputMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap) {
      return nullptr;
    }
    std::unique_ptr<cpec::cpecImg> cpu_diffImgOutputMAPE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap){
       // 创建一个新的cpecImg来存储差异结果
        std::unique_ptr<cpec::cpecImg> diffImg = std::make_unique<cpec::cpecImg>(img1->width, img1->height, 4, std::make_unique<cpec::col4[]>(img1->width * img1->height));

      // 计算每个像素的差异并应用颜色变换
      //float mape = 0.0f;
      const float epsilon = 1e-3f;
      for (int y = 0; y < img1->height; ++y) {
          for (int x = 0; x < img1->width; ++x) {
              cpec::col4& pixel1 = (*img1)(x, y);
              cpec::col4& pixel2 = (*img2)(x, y);
              cpec::col4& pixelDiff = (*diffImg)(x, y);

              const float Lr = 0.2126f * pixel1.r + 0.7152f * pixel1.g + 0.0722f * pixel1.b;
              const float Lt = 0.2126f * pixel2.r + 0.7152f * pixel2.g + 0.0722f * pixel2.b;
              if( !( std::isfinite( Lr * Lr ) ) ||  !( std::isfinite( Lt * Lt ))) continue;

              //mape += std::abs( Lr - Lt ) / ( Lr + epsilon );

              const auto col = colormap( std::abs( Lr - Lt ) / ( Lr + epsilon ));

              pixelDiff.r = col.x;
              pixelDiff.g = col.y; 
              pixelDiff.b = col.z;

              pixelDiff.a = 1.0f;  // 保持透明度为1.0
          }
      }
      return diffImg;
    }
    std::unique_ptr<cpec::cpecImg> gpu_diffImgOutputMAPE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap){
      return nullptr;
    }
    std::unique_ptr<cpec::cpecImg> cpu_diffImgOutputRELMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap){
       // 创建一个新的cpecImg来存储差异结果
        std::unique_ptr<cpec::cpecImg> diffImg = std::make_unique<cpec::cpecImg>(img1->width, img1->height, 4, std::make_unique<cpec::col4[]>(img1->width * img1->height));

      // 计算每个像素的差异并应用颜色变换
      //float relmse = 0.0f;
      const float epsilon = 1e-3f;
      for (int y = 0; y < img1->height; ++y) {
          for (int x = 0; x < img1->width; ++x) {
              cpec::col4& pixel1 = (*img1)(x, y);
              cpec::col4& pixel2 = (*img2)(x, y);
              cpec::col4& pixelDiff = (*diffImg)(x, y);

              const float Lr = 0.2126f * pixel1.r + 0.7152f * pixel1.g + 0.0722f * pixel1.b;
              const float Lt = 0.2126f * pixel2.r + 0.7152f * pixel2.g + 0.0722f * pixel2.b;
              if( !( std::isfinite( Lr * Lr ) ) ||  !( std::isfinite( Lt * Lt ))) continue;

              //relmse += ( Lr - Lt ) * ( Lr - Lt ) / ( Lr * Lr + epsilon );

              const auto col = colormap( ( Lr - Lt ) * ( Lr - Lt ) / ( Lr * Lr + epsilon ));

              pixelDiff.r = col.x;
              pixelDiff.g = col.y; 
              pixelDiff.b = col.z;

              pixelDiff.a = 1.0f;  // 保持透明度为1.0
          }
      }
      return diffImg;
    } 
    std::unique_ptr<cpec::cpecImg> gpu_diffImgOutputRELMSE(const std::unique_ptr<cpec::cpecImg>& img1, const std::unique_ptr<cpec::cpecImg>& img2, const pseudocolor& colormap){
      return nullptr;
    }
}

namespace cpec{
  // 一律看成四通道图像进行载入，低通道的情况下默认使用第一个通道填充后面的通道
  std::unique_ptr<cpecImg> loadImage(const std::string& filepath){
      int width, height, channels;
      unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
      if (!data) {
        std::cerr << "Failed to load image!" << std::endl;
        return nullptr;
      }
      auto imageData = std::make_unique<col4[]>(width * height);

      int idx = 0;
      for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
              // 获取当前像素的RGBA值
              unsigned char* pixel = &data[(i * width + j) * channels];

              double r = (channels >= 1) ? (pixel[0] / 255.0f) : 1.0f;

              double g = (channels >= 2) ? (pixel[1] / 255.0f) : 
                        (channels >= 1) ? (pixel[0] / 255.0f) : 1.0f;

              double b = (channels >= 3) ? (pixel[2] / 255.0f) : 
                        (channels >= 2) ? (pixel[1] / 255.0f) : 
                        (channels >= 1) ? (pixel[0] / 255.0f) : 1.0f;

              double a = (channels >= 4) ? (pixel[3] / 255.0f) : 
                        (channels >= 3) ? (pixel[2] / 255.0f) : 
                        (channels >= 2) ? (pixel[1] / 255.0f) : 
                        (channels >= 1) ? (pixel[0] / 255.0f) : 1.0f;


              // 将像素数据填充到col4对象中
              imageData[idx++] = col4(r, g, b, a);
          }
      }

      // 释放stbi加载的图片数据
      stbi_image_free(data);

      return std::make_unique<cpecImg>(filepath, width, height, channels, std::move(imageData));
  }

  void saveImage(const std::unique_ptr<cpecImg>& img, const std::string& filepath){
        // 获取图像的宽度、高度、通道数以及数据
        int width = img->width;
        int height = img->height;
        int channels = img->channels;
        const auto& data = img->data;

        // 将col4数据转换为unsigned char数组（RGBA格式）
        std::vector<unsigned char> outputData(width * height * 4);

        int idx = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                // 获取col4对象中的颜色值
                const col4& pixel = data[i * width + j];
                
                // 转换为0-255范围的unsigned char值
                outputData[idx++] = static_cast<unsigned char>(pixel.r * 255.0f);  // Red
                outputData[idx++] = static_cast<unsigned char>(pixel.g * 255.0f);  // Green
                outputData[idx++] = static_cast<unsigned char>(pixel.b * 255.0f);  // Blue
                if (channels == 4) {
                    outputData[idx++] = static_cast<unsigned char>(pixel.a * 255.0f);  // Alpha (如果是RGBA格式)
                }
            }
        }

        // 保存图像为PNG文件（你可以根据需要更改为其他格式，如JPG、BMP等）
        if (filepath.substr(filepath.find_last_of(".") + 1) == "png") {
          stbi_write_png(filepath.c_str(), width, height, channels, outputData.data(), width * channels);
        } else if (filepath.substr(filepath.find_last_of(".") + 1) == "jpg" || filepath.substr(filepath.find_last_of(".") + 1) == "jpeg") {
            stbi_write_jpg(filepath.c_str(), width, height, channels, outputData.data(), 100);
        } else if (filepath.substr(filepath.find_last_of(".") + 1) == "bmp") {
            stbi_write_bmp(filepath.c_str(), width, height, channels, outputData.data());
        } else if (filepath.substr(filepath.find_last_of(".") + 1) == "tga") {
            stbi_write_tga(filepath.c_str(), width, height, channels, outputData.data());
        } else if (filepath.substr(filepath.find_last_of(".") + 1) == "hdr") {
            float *floatData = new float[width * height * channels];

            // 将 unsigned char 数据转换为 float 数据（范围 0.0 到 1.0）
            for (int i = 0; i < width * height * channels; ++i) {
                floatData[i] = outputData[i] / 255.0f;
            }

            stbi_write_hdr(filepath.c_str(), width, height, channels, floatData);
        } else {
            std::cerr << "Unsupported file format!" << std::endl;
        }
  }

  double diffValCal( const std::unique_ptr<cpecImg>& img1, const std::unique_ptr<cpecImg>& img2, const std::string& method, int parameter_1, int parameter_2){
        if (img1->isEmpty() || img2->isEmpty()) {
          throw std::invalid_argument("One of the input images is empty");
        }
        
        if(method == "MSE"){
          return parameter_1? cpecInter::cpu_diffValCalMSE(img1, img2) : cpecInter::gpu_diffValCalMSE(img1, img2);
        }else if(method == "PSNR"){
          return parameter_1? cpecInter::cpu_diffValCalPSNR(img1, img2) : cpecInter::gpu_diffValCalPSNR(img1, img2);
        }
        else if(method == "RMSE"){
          return parameter_1? cpecInter::cpu_diffValCalRMSE(img1, img2) : cpecInter::gpu_diffValCalRMSE(img1, img2);
        }
        else if(method == "SSIM"){
          return parameter_1? cpecInter::cpu_diffValCalSSIM(img1, img2, parameter_2) : cpecInter::gpu_diffValCalSSIM(img1, img2, parameter_2);
        }
        else{
          std::cerr << "Unsupported method!" << std::endl;
          return -1;
        }
    }

  std::unique_ptr<cpecImg> diffImgOutput(const std::unique_ptr<cpecImg>& img1, const std::unique_ptr<cpecImg>& img2, const std::string& method, 
   const std::string& output_path, int parameter_1, const std::string& parameter_2, const float& parameter_3, const float& parameter_4){
    if (img1->isEmpty() || img2->isEmpty()) {
          throw std::invalid_argument("One of the input images is empty");
    }

    pseudocolor colormap;
    if (parameter_2 == "JET") {
    colormap = pseudocolor::jet(parameter_3, parameter_4);
    } else if (parameter_2 == "PARULA") {
        colormap = pseudocolor::parula(parameter_3, parameter_4);
    } else if (parameter_2 == "AUTUMN") {
        colormap = pseudocolor::autumn(parameter_3, parameter_4);
    } else if (parameter_2 == "BONE") {
        colormap = pseudocolor::bone(parameter_3, parameter_4);
    } else if (parameter_2 == "COOL") {
        colormap = pseudocolor::cool(parameter_3, parameter_4);
    } else if (parameter_2 == "COPPER") {
        colormap = pseudocolor::copper(parameter_3, parameter_4);
    } else if (parameter_2 == "HOT") {
        colormap = pseudocolor::hot(parameter_3, parameter_4);
    } else if (parameter_2 == "HSV") {
        colormap = pseudocolor::hsv(parameter_3, parameter_4);
    } else if (parameter_2 == "PINK") {
        colormap = pseudocolor::pink(parameter_3, parameter_4);
    } else if (parameter_2 == "SPRING") {
        colormap = pseudocolor::spring(parameter_3, parameter_4);
    } else if (parameter_2 == "SUMMER") {
        colormap = pseudocolor::summer(parameter_3, parameter_4);
    } else if (parameter_2 == "WINTER") {
        colormap = pseudocolor::winter(parameter_3, parameter_4);
    } else {
        std::cerr << "Unsupported colormap!" << std::endl;
        return nullptr;
    }

    std::unique_ptr<cpecImg> diffImg = nullptr;

    if(method == "MSE"){
        diffImg = parameter_1? cpecInter::cpu_diffImgOutputMSE(img1, img2, colormap) : cpecInter::gpu_diffImgOutputMSE(img1, img2, colormap);
    }
    else if(method == "MAPE"){
        diffImg = parameter_1? cpecInter::cpu_diffImgOutputMAPE(img1, img2, colormap) : cpecInter::gpu_diffImgOutputMAPE(img1, img2, colormap);
    }
    else if(method == "RELMSE"){
        diffImg = parameter_1? cpecInter::cpu_diffImgOutputRELMSE(img1, img2, colormap) : cpecInter::gpu_diffImgOutputRELMSE(img1, img2, colormap);
    }
    else{
          std::cerr << "Unsupported method!" << std::endl;
          return nullptr;
    }

    ////////TODO: gpu_diffImgOutput/////////
    if(parameter_1 == 0){
      std::cerr << "gpu diffImg not finished." << std::endl;
          return nullptr;
    }
    ////////TODO: gpu_diffImgOutput/////////


    diffImg->path = output_path;

    // 保存图像到文件
    saveImage(diffImg, output_path);

    // 返回计算好的差异图像
    return diffImg;
  }
}
