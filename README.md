# YoloV8Infer

基于ONNX Runtime和OpenCV的YOLOv8目标检测推理程序。

## 功能特点

- 使用ONNX Runtime进行模型推理
- 支持YOLOv8模型格式
- 实时目标检测
- 支持多种图像格式输入
- 使用spdlog进行日志记录

## 精度对齐
- BGR -> RGB通道转换
- Letterbox resize（保持宽高比 + 灰边填充）
- 边界框坐标还原（inverse letterbox，映射回原图）
- 像素归一化（/255.0）
- 解析输出向量[1,5,13125]并执行NMS，输出(x1, y1, x2, y2, conf, class_id) 
- config.json可设置
--模型加载路径，模型输入尺寸：宽高
--置信度阈值和NMS阈值
--单张测试图片路径
--类别

## 依赖项

- OpenCV 4.x
- ONNX Runtime 1.17.1
- spdlog 1.11.0 或更高版本 (可以通过包管理器安装或手动编译安装)
- C++17 或更高版本
- CMake 3.12 或更高版本
- nlohmann/json (用于JSON配置文件解析)

## 构建说明

1. 安装依赖项：
   - OpenCV: 请修改CMakeLists.txt中的OpenCV_DIR路径
   - ONNX Runtime: 请修改CMakeLists.txt中的ONNXRUNTIME_ROOT路径
   - spdlog: 使用vcpkg安装或修改CMakeLists.txt中的路径

2. 使用CMake构建项目：
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## spdlog安装选项

### 使用包管理器安装spdlog (推荐)

#### Ubuntu/Debian:
```bash
sudo apt-get install libspdlog-dev
```

#### macOS (Homebrew):
```bash
brew install spdlog
```

#### Windows (vcpkg):
```bash
vcpkg install spdlog
```

### 手动安装spdlog

1. 从GitHub下载spdlog源码：
```bash
git clone https://github.com/gabime/spdlog.git
cd spdlog
```

2. 编译安装（可选）：
```bash
mkdir build
cd build
cmake ..
make -j
sudo make install
```

3. 配置环境变量（如果需要）

4. 修改CMakeLists.txt中的SPDLOG_INCLUDE_DIRS路径：
```cmake
set(SPDLOG_INCLUDE_DIRS "/usr/local/include")  # Linux/macOS
# 或者
set(SPDLOG_INCLUDE_DIRS "C:/Program Files/spdlog/include")  # Windows
```

如果您不确定spdlog的安装路径，可以尝试以下方法找到它：
- 检查您解压spdlog源码包的目录（通常是 `spdlog-版本号/include`）
- 如果使用vcpkg安装：`C:/vcpkg/installed/x64-windows/include`
- 如果使用系统安装：`C:/Program Files/spdlog/include`

## nlohmann/json安装选项

### 使用包管理器安装nlohmann/json (推荐)

#### Ubuntu/Debian:
```bash
sudo apt-get install nlohmann-json3-dev
```

#### macOS (Homebrew):
```bash
brew install nlohmann-json
```

#### Windows (vcpkg):
```bash
vcpkg install nlohmann-json
```

### 手动安装nlohmann/json

1. 从GitHub下载nlohmann/json源码：
```bash
git clone https://github.com/nlohmann/json.git
cd json
```

2. 安装（这是一个头文件库，只需要包含头文件）：
```bash
# 复制include目录到您的项目或系统目录
cp -r include/nlohmann /usr/local/include/
```

3. 解压后将包含`include`目录的文件夹路径配置到CMakeLists.txt中

**注意：** nlohmann/json是一个头文件库，不需要编译。只需要确保CMakeLists.txt中的路径正确指向包含nlohmann/json.hpp的目录。

### Visual Studio配置注意事项

如果您在Visual Studio 2022中遇到nlohmann/json找不到的问题，请确保：

1. 检查CMakeLists.txt中的nlohmann/json路径是否正确
2. 确保路径使用正斜杠(/)而不是反斜杠(\)
3. 如果使用vcpkg，请确保已正确集成到Visual Studio中
4. 重新生成CMake缓存：在Visual Studio中选择"项目" -> "删除缓存并重新重新生成"

### 常见问题排查

如果仍然遇到问题，请检查：
1. nlohmann/json目录结构应该是：`include/nlohmann/json.hpp`
2. 头文件应该位于：`include/nlohmann/json.hpp`
3. 在CMakeLists.txt中设置的路径应指向`include`目录的上一级
4. 如果使用本地解压的nlohmann/json，请确保路径配置正确，例如：`D:/install/nljson/json-3.12.0/include`

## 使用方法

```bash
# 使用默认模型和图像
./YoloV8Infer

# 指定图像和模型路径
./YoloV8Infer path/to/image.jpg path/to/model.onnx
```

## 日志系统

本项目使用spdlog作为日志系统，具有以下特性：

- 控制台输出彩色日志
- 文件日志记录到`logs/yolov8_infer.log`
- 支持多种日志级别（debug, info, warn, error）
- 自动时间戳和日志级别标记

日志文件将在程序运行时自动创建在`logs`目录下。

## 配置

配置文件位于`configs/config.json`，可以配置：
- 模型路径
- 检测阈值
- 类别名称
- 输出设置

## 许可证

MIT License