#include "ObjectDetector.h"
#include "JsonConfigManager.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>

int main(int argc, char* argv[]) {
    // Initialize spdlog
    std::shared_ptr<spdlog::logger> console;
    std::shared_ptr<spdlog::logger> file_logger;
    try {
        console = spdlog::stdout_color_mt("console");
        file_logger = spdlog::basic_logger_mt("file_logger", "logs/yolov8_infer.log");
        
        // Create a logger that outputs to both console and file
        std::vector<spdlog::sink_ptr> sinks;
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/yolov8_infer.log", true);
        
        sinks.push_back(console_sink);
        sinks.push_back(file_sink);
        
        auto combined_logger = std::make_shared<spdlog::logger>("combined", sinks.begin(), sinks.end());
        combined_logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(combined_logger);
        
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        return -1;
    }
    
    console->info("Starting YoloV8Infer application");
    
    // Load configuration
    JsonConfigManager config_manager("configs/config.json");
    if (!config_manager.loadConfig()) {
        console->error("Failed to load configuration file");
        return -1;
    }
    
    // Get paths from configuration
    std::string model_path = config_manager.getModelConfig().path;
    std::string image_path = config_manager.getInputConfig().image_path;
    
    // Parse command line arguments
    if (argc > 1) {
        image_path = argv[1];
        console->info("Using image path from command line: {}", image_path);
    }
    if (argc > 2) {
        model_path = argv[2];
        console->info("Using model path from command line: {}", model_path);
    }
    
    try {
        console->info("Loading image: {}", image_path);
        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            console->error("Cannot load image: {}", image_path);
            return -1;
        }
        
        console->info("Image loaded successfully. Size: {}x{}", image.cols, image.rows);
        
        // Initialize detector
        console->info("Initializing ObjectDetector");
        ObjectDetector detector;
        if (!detector.initialize(config_manager)) {
            console->error("Failed to initialize detector from JSON config");
            return -1;
        }
        
        console->info("Model loaded successfully: {}", model_path);
        
        // Perform detection
        console->info("Starting object detection");
        std::vector<DetectionResult> results = detector.detect(image);
        
        // Display results
        console->info("Detection results:");
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            console->info("  [{}] Class: {}, Confidence: {:.2f}, Box: ({}, {}, {}, {})", 
                        i, result.class_id, result.confidence, 
                        result.box.x, result.box.y, result.box.width, result.box.height);
        }
        
        // Show image with detections
        console->info("Displaying results");
        cv::Mat result_image = image.clone();
        detector.drawBoxes(result_image, results);
        cv::imshow("Object Detection Result", result_image);
        cv::waitKey(0);
        
        console->info("Application finished successfully");
        
    }
    catch (const std::exception& e) {
        console->error("Exception: {}", e.what());
        return -1;
    }
    
    return 0;
}