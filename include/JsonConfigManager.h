#pragma once

#include <string>
#include <nlohmann/json.hpp>

// 配置结构体定义
struct ModelConfig {
    std::string path;
    int input_width = 640;
    int input_height = 640;
};

struct DetectionConfig {
    float confidence_threshold = 0.35f;
    float nms_threshold = 0.45f;
};

struct InputConfig {
    std::string image_path;
};

struct ClassesConfig {
    std::vector<std::string> names;
};

class JsonConfigManager {
public:
    explicit JsonConfigManager(const std::string& config_path);
    
    bool loadConfig();
    
    const ModelConfig& getModelConfig() const { return model_config_; }
    const DetectionConfig& getDetectionConfig() const { return detection_config_; }
    const InputConfig& getInputConfig() const { return input_config_; }
    const ClassesConfig& getClassesConfig() const { return classes_config_; }

private:
    std::string config_path_;
    nlohmann::json config_data_;
    
    ModelConfig model_config_;
    DetectionConfig detection_config_;
    InputConfig input_config_;
    ClassesConfig classes_config_;
    
    bool parseModelConfig();
    bool parseDetectionConfig();
    bool parseInputConfig();
    bool parseClassesConfig();
};