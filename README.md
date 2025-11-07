# YoloV8Infer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于ONNX Runtime的YOLOv8目标检测C++实现，支持CPU和GPU推理。

## 功能特性

- 使用C++实现的YOLOv8目标检测
- 通过JSON文件进行配置
- 集成OpenCV进行图像处理
- 使用ONNX Runtime进行模型推理
- 支持CPU和GPU推理
- 跨平台支持（Windows、Linux、macOS）

## 环境要求

- CMake 3.14 或更高版本
- 支持C++17的编译器
- OpenCV 4.x
- ONNX Runtime（CPU或GPU版本）
- spdlog
- nlohmann/json

## 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/YoloV8Infer.git
   cd YoloV8Infer
   ```

2. 安装依赖：
   - OpenCV：从[OpenCV官网](https://opencv.org/)下载并构建
   - ONNX Runtime：从[ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)下载
   - spdlog：从[spdlog GitHub](https://github.com/gabime/spdlog)下载
   - nlohmann/json：从[nlohmann/json GitHub](https://github.com/nlohmann/json)下载

3. 在CMakeLists.txt中配置路径：
   根据您的安装路径更新以下变量：
   ```cmake
   set(OpenCV_DIR "path/to/opencv/build/install")
   set(ONNXRUNTIME_ROOT "path/to/onnxruntime")
   set(SPDLOG_INCLUDE_DIRS "path/to/spdlog/include")
   set(NLOHMANN_JSON_INCLUDE_DIRS "path/to/nlohmann/json/include")
   ```

4. 构建项目：
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## 使用方法

1. 准备YOLOv8的ONNX格式模型。

2. 在`configs/config.json`中配置设置：
   ```json
   {
     "model": {
       "path": "path/to/your/model.onnx",
       "input_width": 640,
       "input_height": 640,
       "device_type": "CPU"  // 或 "GPU"
     },
     "detection": {
       "confidence_threshold": 0.5,
       "nms_threshold": 0.4
     },
     "input": {
       "image_path": "path/to/your/image.jpg"
     },
     "classes": [
       "类别1",
       "类别2"
     ]
   }
   ```

3. 运行应用程序：
   ```bash
   ./YoloV8Infer
   ```

## 性能对比

在我们的测试中，对于较小的模型，由于数据传输开销，CPU推理可能比GPU推理更快。对于较大的模型或批量处理，GPU推理通常能提供更好的性能。

### 测试结果
- CPU推理：约81毫秒
- GPU推理：约1253毫秒（首次运行包含初始化开销）
- [2025-11-07 17:37:03.579] [performance] [info] === PERFORMANCE COMPARISON ===
- [2025-11-07 17:37:03.579] [performance] [info] CPU: 74.21 ms
- [2025-11-07 17:37:03.579] [performance] [info] GPU: 7.05 ms
- [2025-11-07 17:37:03.579] [performance] [info] GPU is 10.53x faster than CPU

优化GPU性能的方法：
1. 对多张图片使用批量处理
2. 确保模型足够大以从GPU并行化中受益
3. 在代码中调整CUDA提供程序选项

## 项目结构

```
YoloV8Infer/
├── src/                 # 源代码
├── include/             # 头文件
├── configs/             # 配置文件
├── tests/               # 单元测试
├── CMakeLists.txt       # CMake构建脚本
├── README.md            # 项目文档
└── LICENSE              # 许可证文件
```

## 贡献

欢迎贡献！请随时提交Pull Request。

## 许可证

该项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件了解详情。