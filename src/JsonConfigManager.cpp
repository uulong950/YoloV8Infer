#include "JsonConfigManager.h"
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

JsonConfigManager::JsonConfigManager(const std::string& config_path) 
    : config_path_(config_path) {}

bool JsonConfigManager::loadConfig() {
    try {
        std::ifstream file(config_path_);
        if (!file.is_open()) {
            spdlog::error("Failed to open config file: {}", config_path_);
            return false;
        }
        
        file >> config_data_;
        file.close();
        
        // 解析各个配置部分
        if (!parseModelConfig()) {
            return false;
        }
        
        if (!parseDetectionConfig()) {
            return false;
        }
        
        if (!parseInputConfig()) {
            return false;
        }
        
        if (!parseClassesConfig()) {
            return false;
        }
        
        spdlog::info("Configuration loaded successfully from {}", config_path_);
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to load config: {}", e.what());
        return false;
    }
}

bool JsonConfigManager::parseModelConfig() {
    try {
        if (config_data_.contains("model")) {
            const auto& model = config_data_["model"];
            if (model.contains("path")) {
                model_config_.path = model["path"].get<std::string>();
            }
            if (model.contains("input_width")) {
                model_config_.input_width = model["input_width"].get<int>();
            }
            if (model.contains("input_height")) {
                model_config_.input_height = model["input_height"].get<int>();
            }
            if (model.contains("device_type")) {
                model_config_.device_type = model["device_type"].get<std::string>();
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to parse model config: {}", e.what());
        return false;
    }
}

bool JsonConfigManager::parseDetectionConfig() {
    try {
        if (config_data_.contains("detection")) {
            const auto& detection = config_data_["detection"];
            if (detection.contains("confidence_threshold")) {
                detection_config_.confidence_threshold = detection["confidence_threshold"].get<float>();
            }
            if (detection.contains("nms_threshold")) {
                detection_config_.nms_threshold = detection["nms_threshold"].get<float>();
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to parse detection config: {}", e.what());
        return false;
    }
}

bool JsonConfigManager::parseInputConfig() {
    try {
        if (config_data_.contains("input")) {
            const auto& input = config_data_["input"];
            if (input.contains("image_path")) {
                input_config_.image_path = input["image_path"].get<std::string>();
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to parse input config: {}", e.what());
        return false;
    }
}

bool JsonConfigManager::parseClassesConfig() {
    try {
        if (config_data_.contains("classes")) {
            const auto& classes = config_data_["classes"];
            if (classes.is_array()) {
                for (const auto& class_name : classes) {
                    if (class_name.is_string()) {
                        classes_config_.names.push_back(class_name.get<std::string>());
                    }
                }
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to parse classes config: {}", e.what());
        return false;
    }
}