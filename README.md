# CPEC 使用指南 (Update：v0.1.0-alpha)

## TODO

1. 增加输出对比图的gpu版本。
2. 根据输入图片自动选择cpu/gpu方案。
3. 增加指定输出图片路径/格式功能。
4. 增加SSIM blocksize参数。
5. 减小gpu版本精度误差。
6. 增加GUI，实时查看比对图片与比对值。

## 使用前须知

**1. 使用CMakeLists维护，如有需要请使用```git clone --recursive```克隆后，重新编译静态库。**

**2. 本项目采用cuda v12.6构建。**

**3. 目前gpu版本正在开发中，仅支持8k及以下图片输入。**

## 参数

**请保证输入的图片分辨率一致！！！**
**使用gpu版本请提前安装CUDA！！！**
**4K以上分辨率图片对比推荐使用gpu**

## 输出计算值
```cpec.exe [reference_filename] [target_filename] [diffval] [method] [cpu|gpu]```

例：```cpec.exe ./1.png ./2.jpg diffval MSE cpu```

其中，method有：

-  MSE （均方误差）：衡量对比图片之间差异的平方的平均值，值越小表示误差越小。

-  RMSE（均方根误差）：MSE的平方根，与原始数据单位一致，更直观反映误差大小。
-  PSNR（峰值信噪比）：值越大表示失真越小，质量越高。
-  SSIM（结构相似性）：评估两幅图像在亮度、对比度和结构上的相似性，值越接近1表示越相似。

目前，v0.1.0-alpha版本SSIM的 blocksize 不支持参数输入，默认为8。
MSE、RMSE以及PSNR的cpu版本和gpu版本输出结果一致，没有误差；但SSIM有$<10^{-4}$ 的精度误差。


## 输出比对图

**比对图目前目前只支持cpu版本！！！**

```cpec.exe [reference_filename] [target_filename] [diffimg] [method] [cpu] [colormap_style] [min_value] [max_value]```

例：```cpec.exe ./1.png ./2.jpg diffimg MSE cpu PINK 0.0 0.5```

其中，method有：

- MSE （均方误差）：衡量对比图片之间差异的平方的平均值，值越小表示误差越小。
- MAPE（平均绝对百分比误差）：衡量图片之间的百分比误差的平均值，值越小表示误差越小，常用于评估相对误差。
- RELMSE（相对均方误差）：将均方误差（MSE）除以真实值的平方和，用于标准化误差，便于不同数据集间的比较。

其中，colormap_style 有：

```colormaps = ["JET", "PARULA", "AUTUMN", "BONE", "COOL", "COPPER", "HOT", "HSV", "PINK", "SPRING", "SUMMER", "WINTER"]```

min_value：表示数据的最小值，对应颜色映射的起点。

max_value：表示数据的最大值，对应颜色映射的终点。

数据在 min_value 和 max_value 之间线性映射到颜色条上，超出范围的数据通常会被截断或循环处理。
