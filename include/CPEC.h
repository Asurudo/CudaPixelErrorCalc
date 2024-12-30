#ifndef CPEC_H
#define CPEC_H

#include <functional>

namespace cpec{
    class col4{
    public:
        double r,g,b,a;
        col4(double r,double g,double b,double a):r(r),g(g),b(b),a(a){}
        col4(double r,double g,double b):r(r),g(g),b(b),a(1.0f){}
        col4():r(0.0f),g(0.0f),b(0.0f),a(1.0f){}
    };
    
    // 图片的读入，输出，基本信息储存（高度宽度，rgb值等）
    class cpecImg{
    public:
        std::string path;
        int width;
        int height;
        int channels;
        std::unique_ptr<col4[]> data;
        cpecImg(){
            width = 0;
            height = 0;
            channels = 0;
            path = "";
            data = nullptr;
        }
        cpecImg(int width, int height, int channels, std::unique_ptr<col4[]> data):width(width),height(height),channels(channels), data(std::move(data)){
            
        }
        cpecImg(std::string path, int width, int height, int channels, std::unique_ptr<col4[]> data):path(path), width(width),height(height),channels(channels), data(std::move(data)){
            
        }
        ~cpecImg(){
            
        }
        bool isEmpty(){
            return data == nullptr;
        }
        col4& operator()(int x, int y){
            if(y * width + x >= width * height){
                throw std::runtime_error("Out of range");
            }
            return data[y * width + x];
        }
    };

    // 根据图片的后缀名，利用stb_image库读入图片，返回一个cpecImg对象的C++ style指针
    std::unique_ptr<cpecImg> loadImage(const std::string& filepath);
    // 将cpecImg对象保存为图片文件
    void saveImage(const std::unique_ptr<cpecImg>& img, const std::string& filepath);

    // 两张图片的各种相似度Error值计算（允许传入参数或者传入一个比较函数）
    double diffValCal( const std::unique_ptr<cpecImg>& img1, const std::unique_ptr<cpecImg>& img2, const std::string& method = "MSE", 
                       int parameter_1 = 1, int parameter_2 = 8);
    //double diffValCal( const std::unique_ptr<cpecImg>& img1, const std::unique_ptr<cpecImg>& img2, 
    //                      std::function<double(const std::unique_ptr<cpecImg>&, const std::unique_ptr<cpecImg>&)> customFunc);
    
    // 输出两张图片差异图
    std::unique_ptr<cpecImg> diffImgOutput(const std::unique_ptr<cpecImg>& img1, const std::unique_ptr<cpecImg>& img2, 
                                           const std::string& method = "MSE", 
                                           const std::string& output_path = "./diffImgOutput.png", 
                                           int parameter_1 = 1, const std::string& parameter_2 = "PARULA",
                                           const float& parameter_3 = 0.f, const float& parameter_4 = 10.f);
} 


#endif