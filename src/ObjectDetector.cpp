#include "ObjectDetector.h"
#include "JsonConfigManager.h"
#include <algorithm>
#include <cmath>

ObjectDetector::ObjectDetector() 
    : confidence_threshold_(0.0f)
    , nms_threshold_(0.0f)
    , input_width_(0)
    , input_height_(0) {
    // 类别名称将从JSON配置中加载
    spdlog::info("ObjectDetector initialized");
}

ObjectDetector::~ObjectDetector() = default;

bool ObjectDetector::initialize(JsonConfigManager& config_manager) {
    try {
        // 从JSON配置获取参数
        const auto& model_config = config_manager.getModelConfig();
        const auto& detection_config = config_manager.getDetectionConfig();
        const auto& classes_config = config_manager.getClassesConfig();
        
        // 设置模型参数
        input_width_ = model_config.input_width;
        input_height_ = model_config.input_height;
        confidence_threshold_ = detection_config.confidence_threshold;
        nms_threshold_ = detection_config.nms_threshold;
        device_type_ = model_config.device_type; // 设置设备类型
        
        // 设置类别名称
        if (!classes_config.names.empty()) {
            class_names_ = classes_config.names;
        } else {
            // 默认类别名称
            class_names_ = {"face"};
        }
        
        spdlog::info("Initializing ObjectDetector from JSON config");
        spdlog::info("Model path: {}", model_config.path);
        spdlog::info("Input size: {}x{}", input_width_, input_height_);
        spdlog::info("Confidence threshold: {}", confidence_threshold_);
        spdlog::info("NMS threshold: {}", nms_threshold_);
        spdlog::info("Device type: {}", device_type_);
        spdlog::info("Number of classes: {}", class_names_.size());
        
        // 调用基础初始化方法
        return initialize(model_config.path);
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to initialize from JSON config: {}", e.what());
        return false;
    }
}

bool ObjectDetector::initialize(const std::string& model_path) {
    try {
        spdlog::info("Initializing ObjectDetector with model: {}", model_path);
        
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ObjectDetector");
        
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        // 根据设备类型配置推理提供者
        if (device_type_ == "GPU") {
            // Enable CUDA provider for GPU inference
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;  // Use the first GPU
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            // 添加更多优化选项
            // 如果需要更高的性能，可以尝试不同的cudnn_conv_algo_search选项：
            // OrtCudnnConvAlgoSearchExhaustive (默认) - 最优化但可能较慢
            // OrtCudnnConvAlgoSearchHeuristic - 启发式搜索，较快但可能不是最优
            // OrtCudnnConvAlgoSearchNone - 不搜索，最快但可能不是最优
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            spdlog::info("Using GPU for inference with CUDA provider");
        } else {
            // CPU inference - no additional providers needed
            spdlog::info("Using CPU for inference");
        }
        
        // Convert string to wide string for ONNX Runtime
        std::wstring w_model_path(model_path.begin(), model_path.end());
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, w_model_path.c_str(), session_options);
        
        // Get input and output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names
        size_t num_input_nodes = session_->GetInputCount();
        input_node_names_.resize(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_node_names_[i] = input_name.get();
        }
        
        // Output names
        size_t num_output_nodes = session_->GetOutputCount();
        output_node_names_.resize(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_node_names_[i] = output_name.get();
        }
        
        spdlog::info("Model loaded successfully. Input nodes: {}, Output nodes: {}", 
                            num_input_nodes, num_output_nodes);
        
        return true;
    }
    catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime Exception: {}", e.what());
        return false;
    }
    catch (const std::exception& e) {
        spdlog::error("Standard Exception: {}", e.what());
        return false;
    }
}

std::vector<DetectionResult> ObjectDetector::detect(const cv::Mat& image) {
    std::vector<DetectionResult> results;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        spdlog::info("Starting detection on image ({}x{})", image.cols, image.rows);
        
        // Letterbox preprocessing
        cv::Mat letterbox_image = letterboxResize(image, cv::Size(input_width_, input_height_));
        
        // Convert to blob
        cv::Mat blob;
        cv::dnn::blobFromImage(letterbox_image, blob, 1.0 / 255.0, cv::Size(),
            cv::Scalar(0, 0, 0), true, false);

        // Prepare input tensor
        std::array<int64_t, 4> input_shape{ 1, 3, input_height_, input_width_ };
        auto input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
            (float*)blob.data, blob.total(), input_shape.data(), input_shape.size());

        // 🔧 转换 std::string -> const char*
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        for (const auto& name : input_node_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : output_node_names_) {
            output_names_cstr.push_back(name.c_str());
        }

        // 正确调用 Run
        auto output_tensors = session_->Run(
            Ort::RunOptions{ nullptr },
            input_names_cstr.data(),
            &input_tensor,
            input_names_cstr.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
        
        // Process output [1, 5, 13125]
        float* raw_output = output_tensors.front().GetTensorMutableData<float>();
        Ort::TensorTypeAndShapeInfo output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_info.GetShape();
        
        if (output_dims.size() != 3 || output_dims[0] != 1 || output_dims[1] != 5) {
            spdlog::error("Output tensor format error! Expected [1, 5, N], got [{}, {}, {}]", 
                                output_dims.size(), output_dims[0], output_dims[1]);
            return results;
        }
        
        int num_anchors = static_cast<int>(output_dims[2]);
        // 使用配置文件中的类别数量，如果未设置则默认为1
        int num_classes = static_cast<int>(class_names_.size());
        if (num_classes == 0) {
            num_classes = 1;  // 默认类别数
        }
        
        spdlog::debug("Processing {} anchors with {} classes", num_anchors, num_classes);
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        // 计算Letterbox的缩放比例和偏移量
        float scale = std::min(static_cast<float>(input_width_) / image.cols, 
                              static_cast<float>(input_height_) / image.rows);
        cv::Size new_size(static_cast<int>(image.cols * scale), static_cast<float>(image.rows * scale));
        int top = (input_height_ - new_size.height) / 2;
        int left = (input_width_ - new_size.width) / 2;
        
        // 计算坐标转换参数
        float scale_x = static_cast<float>(image.cols) / new_size.width;
        float scale_y = static_cast<float>(image.rows) / new_size.height;
        
        int valid_detections = 0;
        for (int i = 0; i < num_anchors; ++i) {
            float cx = raw_output[0 * num_anchors + i];
            float cy = raw_output[1 * num_anchors + i];
            float w = raw_output[2 * num_anchors + i];
            float h = raw_output[3 * num_anchors + i];
            
            // Find maximum class confidence
            float max_conf = -1.0f;
            int best_class = -1;
            for (int c = 0; c < num_classes; ++c) {
                float score = raw_output[(4 + c) * num_anchors + i];
                if (score > max_conf) {
                    max_conf = score;
                    best_class = c;
                }
            }
            
            if (max_conf > confidence_threshold_ && best_class >= 0) {
                // Convert to original image coordinates (考虑Letterbox偏移)
                float x1 = ((cx - w * 0.5f) - left) * scale_x;
                float y1 = ((cy - h * 0.5f) - top) * scale_y;
                float x2 = ((cx + w * 0.5f) - left) * scale_x;
                float y2 = ((cy + h * 0.5f) - top) * scale_y;
                
                // Clip to image boundaries
                int left_clip = static_cast<int>(std::max(0.0f, x1));
                int top_clip = static_cast<int>(std::max(0.0f, y1));
                int right_clip = static_cast<int>(std::min(static_cast<float>(image.cols), x2));
                int bottom_clip = static_cast<int>(std::min(static_cast<float>(image.rows), y2));
                
                if (right_clip > left_clip && bottom_clip > top_clip) {
                    boxes.emplace_back(left_clip, top_clip, right_clip - left_clip, bottom_clip - top_clip);
                    confidences.push_back(max_conf);
                    class_ids.push_back(best_class);
                    valid_detections++;
                }
            }
        }
        
        spdlog::debug("Found {} valid detections before NMS", valid_detections);
        
        // Apply NMS
        std::vector<int> indices = nmsBoxes(boxes, confidences);
        
        // Prepare final results
        for (int idx : indices) {
            DetectionResult result;
            result.box = boxes[idx];
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            results.push_back(result);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        spdlog::info("Detection completed in {} ms. Found {} objects", duration.count(), results.size());
        
    }
    catch (const Ort::Exception& e) {
        spdlog::error("ONNX Runtime Exception during inference: {}", e.what());
    }
    catch (const cv::Exception& e) {
        spdlog::error("OpenCV Exception during inference: {}", e.what());
    }
    catch (const std::exception& e) {
        spdlog::error("Standard Exception during inference: {}", e.what());
    }
    
    return results;
}

std::vector<int> ObjectDetector::nmsBoxes(const std::vector<cv::Rect>& boxes, 
                                         const std::vector<float>& confidences) {
    std::vector<int> indices;
    
    // Get candidates above confidence threshold
    std::vector<int> candidates;
    for (size_t i = 0; i < confidences.size(); ++i) {
        if (confidences[i] >= confidence_threshold_) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    // Sort by confidence
    std::sort(candidates.begin(), candidates.end(),
              [&confidences](int a, int b) {
                  return confidences[a] > confidences[b];
              });
    
    // NMS
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        
        int curr_idx = candidates[i];
        indices.push_back(curr_idx);
        
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            
            int next_idx = candidates[j];
            cv::Rect intersection = boxes[curr_idx] & boxes[next_idx];
            cv::Rect union_rect = boxes[curr_idx] | boxes[next_idx];
            
            float iou = 0.0f;
            if (union_rect.area() > 0) {
                iou = static_cast<float>(intersection.area()) / union_rect.area();
            }
            
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return indices;
}

void ObjectDetector::setConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
}

void ObjectDetector::setNMSThreshold(float threshold) {
    nms_threshold_ = threshold;
}

void ObjectDetector::drawBoxes(cv::Mat& image, const std::vector<DetectionResult>& detections) {
    for (const auto& detection : detections) {
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("%s: %.2f", class_names_[detection.class_id].c_str(), detection.confidence);
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point top_left = cv::Point(detection.box.x, std::max(detection.box.y - label_size.height - 10, 0));
        cv::Point bottom_right = cv::Point(detection.box.x + label_size.width, detection.box.y);
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(detection.box.x, std::max(detection.box.y - 5, 10)),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

cv::Mat ObjectDetector::letterboxResize(const cv::Mat& image, cv::Size target_size, cv::Scalar fill_color) {
    // 计算缩放比例，保持宽高比
    float scale = std::min(static_cast<float>(target_size.width) / image.cols, 
                          static_cast<float>(target_size.height) / image.rows);
    
    // 计算新尺寸
    cv::Size new_size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale));
    
    // 创建目标尺寸的填充图像
    cv::Mat resized_image(target_size, CV_8UC3, fill_color);
    
    // 计算偏移量以居中放置图像
    int top = (target_size.height - new_size.height) / 2;
    int left = (target_size.width - new_size.width) / 2;
    
    // 调整图像大小
    cv::Mat scaled_image;
    cv::resize(image, scaled_image, new_size, 0, 0, cv::INTER_LINEAR);
    
    // 将调整大小后的图像放置在填充图像的中心
    scaled_image.copyTo(resized_image(cv::Rect(left, top, new_size.width, new_size.height)));
    
    return resized_image;
}