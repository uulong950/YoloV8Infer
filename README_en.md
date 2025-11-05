# YoloV8Infer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A C++ implementation of YOLOv8 object detection with ONNX Runtime.

## Features

- YOLOv8 object detection implementation in C++
- Configuration via JSON files
- Integration with OpenCV for image processing
- ONNX Runtime for model inference
- Cross-platform support (Windows, Linux, macOS)

## Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler
- OpenCV 4.x
- ONNX Runtime
- spdlog
- nlohmann/json

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/YoloV8Infer.git
   cd YoloV8Infer
   ```

2. Install dependencies:
   - OpenCV: Download and build from [OpenCV official website](https://opencv.org/)
   - ONNX Runtime: Download from [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
   - spdlog: Download from [spdlog GitHub](https://github.com/gabime/spdlog)
   - nlohmann/json: Download from [nlohmann/json GitHub](https://github.com/nlohmann/json)

3. Configure paths in CMakeLists.txt:
   Update the following variables according to your installation paths:
   ```cmake
   set(OpenCV_DIR "path/to/opencv/build/install")
   set(ONNXRUNTIME_ROOT "path/to/onnxruntime")
   set(SPDLOG_INCLUDE_DIRS "path/to/spdlog/include")
   set(NLOHMANN_JSON_INCLUDE_DIRS "path/to/nlohmann/json/include")
   ```

4. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## Usage

1. Prepare your YOLOv8 model in ONNX format.

2. Configure the settings in `configs/config.json`:
   ```json
   {
     "model": {
       "path": "path/to/your/model.onnx",
       "input_width": 640,
       "input_height": 640
     },
     "detection": {
       "confidence_threshold": 0.5,
       "nms_threshold": 0.4
     },
     "input": {
       "image_path": "path/to/your/image.jpg"
     }
   }
   ```

3. Run the application:
   ```bash
   ./YoloV8Infer
   ```

## Project Structure

```
YoloV8Infer/
├── src/                 # Source code
├── include/             # Header files
├── configs/             # Configuration files
├── tests/               # Unit tests
├── CMakeLists.txt       # CMake build script
├── README.md            # Project documentation
└── LICENSE              # License file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.