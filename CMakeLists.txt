# CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(YoloV8Infer LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===================
# Windows: UTF-8
# ===================
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ===================
# OpenCV 
# ===================
set(OpenCV_DIR "D:/install/opencv/build/install")
find_package(OpenCV REQUIRED)
set(OPENCV_INCLUDE_DIRS "D:/install/opencv/build/install/include")
if(NOT EXISTS "${OPENCV_INCLUDE_DIRS}/opencv2/opencv.hpp")
    message(FATAL_ERROR "OpenCV header not found at ${OPENCV_INCLUDE_DIRS}")
endif()

# ===================
# ONNX Runtime  
# ===================
set(ONNXRUNTIME_ROOT "D:/install/onnxruntime/onnxruntime-win-x64-1.17.1")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime header not found at ${ONNXRUNTIME_INCLUDE_DIRS}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARIES}")
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIBRARIES}")
endif()

# ===================
# spdlog 
# ===================
# 设置spdlog路径 - 请根据您的实际安装路径修改
set(SPDLOG_INCLUDE_DIRS "D:/install/spdlog/spdlog-1.16.0/include")

# 检查spdlog头文件是否存在
if(NOT EXISTS "${SPDLOG_INCLUDE_DIRS}/spdlog/spdlog.h")
    message(WARNING "spdlog header not found at ${SPDLOG_INCLUDE_DIRS}")
    message(WARNING "Current working directory: ${CMAKE_CURRENT_SOURCE_DIR}")
    message(WARNING "Please check the following common installation paths:")
    message(WARNING "  - If using vcpkg: C:/vcpkg/installed/x64-windows/include")
    message(WARNING "  - If using system installation: C:/Program Files/spdlog/include")
    message(WARNING "  - If manually downloaded: check your extraction directory")
    message(WARNING "You need to adjust SPDLOG_INCLUDE_DIRS in CMakeLists.txt to point to the correct path")
else()
    message(STATUS "Found spdlog at ${SPDLOG_INCLUDE_DIRS}")
endif()

# ===================
# nlohmann/json
# ===================
# 使用本地解压的nlohmann/json
set(NLOHMANN_JSON_INCLUDE_DIRS "D:/install/nljson/json-3.12.0/include")

if(NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIRS}/nlohmann/json.hpp")
    message(FATAL_ERROR "nlohmann/json header not found at ${NLOHMANN_JSON_INCLUDE_DIRS}")
    message(FATAL_ERROR "Please check the path to nlohmann/json - it should point to the directory containing the 'include' folder")
else()
    message(STATUS "Found nlohmann/json at ${NLOHMANN_JSON_INCLUDE_DIRS}")
endif()

# ===================
# Core Library
# ===================
add_library(YoloDetector STATIC
    src/ObjectDetector.cpp
    src/JsonConfigManager.cpp
)
target_include_directories(YoloDetector PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# 添加nlohmann/json包含目录
target_include_directories(YoloDetector PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})

target_link_libraries(YoloDetector PRIVATE 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)
target_compile_features(YoloDetector PUBLIC cxx_std_17)

# ===================
# Main Executable
# ===================
add_executable(${PROJECT_NAME}
    src/main.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${SPDLOG_INCLUDE_DIRS}
)
if(EXISTS "${NLOHMANN_JSON_INCLUDE_DIRS}/nlohmann/json.hpp")
    target_include_directories(${PROJECT_NAME} PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE YoloDetector)

# ===================
# Copy Config File
# ===================
# Automatically copy config file to build directory
configure_file(configs/config.json ${CMAKE_CURRENT_BINARY_DIR}/configs/config.json COPYONLY)

# ===================
# Load Test 
# ===================
enable_testing()
add_subdirectory(tests)