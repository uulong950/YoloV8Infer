#include "ObjectDetector.h"
#include "JsonConfigManager.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <numeric>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

// 性能测试函数
double runPerformanceTest(ObjectDetector& detector, const cv::Mat& image, int iterations = 10) {
    std::vector<double> times;
    
    // 预热运行几次
    for (int i = 0; i < 3; ++i) {
        detector.detect(image);
    }
    
    // 实际计时运行
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector.detect(image);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // 转换为毫秒
    }
    
    // 计算平均时间（排除前几次运行）
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    return avg_time;
}

int main(int argc, char* argv[]) {
    // 初始化日志
    try {
        std::vector<spdlog::sink_ptr> sinks;
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/performance_test.log", true);
        
        sinks.push_back(console_sink);
        sinks.push_back(file_sink);
        
        auto combined_logger = std::make_shared<spdlog::logger>("performance", sinks.begin(), sinks.end());
        combined_logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(combined_logger);
        
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        return -1;
    }
    
    spdlog::info("Starting performance test");
    
    if (argc < 2) {
        spdlog::error("Usage: {} <image_path> [model_path]", argv[0]);
        return -1;
    }
    
    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        spdlog::error("Cannot load image: {}", image_path);
        return -1;
    }
    
    spdlog::info("Image loaded successfully. Size: {}x{}", image.cols, image.rows);
    
    // 测试CPU推理性能
    spdlog::info("Testing CPU inference performance...");
    // 尝试不同的配置文件路径
    //std::string cpu_config_path = "../configs/cpu_config.json";
    std::string cpu_config_path = "cpu_config.json";
    JsonConfigManager cpu_config(cpu_config_path);
    if (!cpu_config.loadConfig()) {
        // 如果相对路径失败，尝试绝对路径
        cpu_config_path = "D:/zxlong/workspace/pro/YoloV8Infer/configs/cpu_config.json";
        JsonConfigManager cpu_config_abs(cpu_config_path);
        if (!cpu_config_abs.loadConfig()) {
            spdlog::error("Failed to load CPU configuration file from both relative and absolute paths");
            return -1;
        }
        cpu_config = std::move(cpu_config_abs);
    }
    
    ObjectDetector cpu_detector;
    if (!cpu_detector.initialize(cpu_config)) {
        spdlog::error("Failed to initialize CPU detector");
        return -1;
    }
    
    double cpu_avg_time = runPerformanceTest(cpu_detector, image, 20);
    spdlog::info("CPU Average inference time over 20 runs: {:.2f} ms", cpu_avg_time);
    
    // 测试GPU推理性能
    spdlog::info("Testing GPU inference performance...");
    // 尝试不同的配置文件路径
    //std::string gpu_config_path = "../configs/gpu_config.json";
    std::string gpu_config_path = "gpu_config.json";
    JsonConfigManager gpu_config(gpu_config_path);
    if (!gpu_config.loadConfig()) {
        // 如果相对路径失败，尝试绝对路径
        gpu_config_path = "D:/zxlong/workspace/pro/YoloV8Infer/configs/gpu_config.json";
        JsonConfigManager gpu_config_abs(gpu_config_path);
        if (!gpu_config_abs.loadConfig()) {
            spdlog::error("Failed to load GPU configuration file from both relative and absolute paths");
            return -1;
        }
        gpu_config = std::move(gpu_config_abs);
    }
    
    ObjectDetector gpu_detector;
    if (!gpu_detector.initialize(gpu_config)) {
        spdlog::error("Failed to initialize GPU detector");
        return -1;
    }
    
    double gpu_avg_time = runPerformanceTest(gpu_detector, image, 20);
    spdlog::info("GPU Average inference time over 20 runs: {:.2f} ms", gpu_avg_time);
    
    // 输出比较结果
    spdlog::info("=== PERFORMANCE COMPARISON ===");
    spdlog::info("CPU: {:.2f} ms", cpu_avg_time);
    spdlog::info("GPU: {:.2f} ms", gpu_avg_time);
    
    if (gpu_avg_time < cpu_avg_time) {
        double speedup = cpu_avg_time / gpu_avg_time;
        spdlog::info("GPU is {:.2f}x faster than CPU", speedup);
    } else {
        double slowdown = gpu_avg_time / cpu_avg_time;
        spdlog::info("GPU is {:.2f}x slower than CPU", slowdown);
    }
    
    return 0;
}