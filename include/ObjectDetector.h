#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

// 前向声明JSON配置管理器
class JsonConfigManager;

struct DetectionResult {
    cv::Rect box;
    int class_id;
    float confidence;
};

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();
    
    // 使用JSON配置初始化
    bool initialize(JsonConfigManager& config_manager);
    
    // 传统初始化方法
    bool initialize(const std::string& model_path);
    
    std::vector<DetectionResult> detect(const cv::Mat& image);
    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void drawBoxes(cv::Mat& image, const std::vector<DetectionResult>& detections);
    
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<std::string> class_names_;
    float confidence_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
    
    // spdlog logger
    std::shared_ptr<spdlog::logger> console_logger_;
    
    std::vector<int> nmsBoxes(const std::vector<cv::Rect>& boxes, 
                             const std::vector<float>& confidences);
};

#endif // OBJECT_DETECTOR_H