#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "../include/core/graph.h"
#include "../include/core/tensor.h"
#include "../include/utils/model_loader.h"
#include "../include/utils/profiler.h"
#include "../include/utils/logger.h"
#include "../include/optimizations/quantization.h"

using namespace deep_engine;

// YOLO specific structures
struct Detection {
    float x, y, w, h;  // Bounding box
    float confidence;   // Object confidence
    int class_id;       // Class ID
    float class_prob;   // Class probability
};

// COCO class names
std::vector<std::string> load_coco_names(const std::string& path) {
    std::vector<std::string> class_names;
    std::ifstream file(path);
    std::string line;
    
    while (std::getline(file, line)) {
        class_names.push_back(line);
    }
    
    if (class_names.empty()) {
        // Default COCO classes if file not found
        class_names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };
    }
    
    return class_names;
}

// Preprocess image for YOLO
Tensor preprocess_yolo_image(const cv::Mat& img, int target_size = 640) {
    // Resize with padding to maintain aspect ratio
    int orig_h = img.rows;
    int orig_w = img.cols;
    float scale = std::min(float(target_size) / orig_h, float(target_size) / orig_w);
    
    int new_h = int(orig_h * scale);
    int new_w = int(orig_w * scale);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    // Create padded image
    cv::Mat padded = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    int pad_h = (target_size - new_h) / 2;
    int pad_w = (target_size - new_w) / 2;
    
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    // Convert to RGB and normalize
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);
    
    // Create tensor in NCHW format
    Tensor input({1, 3, target_size, target_size}, DataType::FP32);
    float* data = input.data<float>();
    
    // Copy data with channel-first format
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_size; ++h) {
            for (int w = 0; w < target_size; ++w) {
                data[c * target_size * target_size + h * target_size + w] = 
                    padded.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    return input;
}

// Non-Maximum Suppression
std::vector<Detection> nms(std::vector<Detection>& detections, float nms_threshold = 0.45) {
    // Sort by confidence
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        // Suppress overlapping boxes
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Calculate IoU
            float x1 = std::max(detections[i].x - detections[i].w/2, 
                               detections[j].x - detections[j].w/2);
            float y1 = std::max(detections[i].y - detections[i].h/2,
                               detections[j].y - detections[j].h/2);
            float x2 = std::min(detections[i].x + detections[i].w/2,
                               detections[j].x + detections[j].w/2);
            float y2 = std::min(detections[i].y + detections[i].h/2,
                               detections[j].y + detections[j].h/2);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float union_area = detections[i].w * detections[i].h + 
                              detections[j].w * detections[j].h - intersection;
            
            float iou = intersection / union_area;
            
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// Process YOLO output
std::vector<Detection> process_yolo_output(const Tensor& output, 
                                         float conf_threshold = 0.25,
                                         float nms_threshold = 0.45,
                                         int num_classes = 80) {
    std::vector<Detection> detections;
    
    // YOLOv5 output format: [1, 25200, 85] for 640x640 input
    // 85 = 5 (x, y, w, h, obj_conf) + 80 (class probabilities)
    const float* data = output.data<float>();
    int num_detections = output.shape()[1];
    int detection_size = output.shape()[2];
    
    for (int i = 0; i < num_detections; ++i) {
        const float* detection = data + i * detection_size;
        
        float obj_conf = detection[4];
        if (obj_conf < conf_threshold) continue;
        
        // Find best class
        int best_class = 0;
        float best_prob = 0;
        for (int c = 0; c < num_classes; ++c) {
            float prob = detection[5 + c];
            if (prob > best_prob) {
                best_prob = prob;
                best_class = c;
            }
        }
        
        float confidence = obj_conf * best_prob;
        if (confidence < conf_threshold) continue;
        
        Detection det;
        det.x = detection[0];
        det.y = detection[1];
        det.w = detection[2];
        det.h = detection[3];
        det.confidence = confidence;
        det.class_id = best_class;
        det.class_prob = best_prob;
        
        detections.push_back(det);
    }
    
    // Apply NMS
    return nms(detections, nms_threshold);
}

// Draw detections on image
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_names) {
    // Generate random colors for each class
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < 80; ++i) {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }
    
    for (const auto& det : detections) {
        // Convert from normalized coordinates to pixel coordinates
        int x1 = int((det.x - det.w/2) * img.cols);
        int y1 = int((det.y - det.h/2) * img.rows);
        int x2 = int((det.x + det.w/2) * img.cols);
        int y2 = int((det.y + det.h/2) * img.rows);
        
        // Draw bounding box
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), 
                     colors[det.class_id], 2);
        
        // Draw label
        std::string label = class_names[det.class_id] + 
                           " " + std::to_string(int(det.confidence * 100)) + "%";
        
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                            0.5, 1, &baseline);
        
        // Draw label background
        cv::rectangle(img, cv::Point(x1, y1 - label_size.height - 10),
                     cv::Point(x1 + label_size.width, y1),
                     colors[det.class_id], cv::FILLED);
        
        // Draw label text
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input> [options]" << std::endl;
        std::cerr << "Input can be:" << std::endl;
        std::cerr << "  - Image file path" << std::endl;
        std::cerr << "  - Video file path" << std::endl;
        std::cerr << "  - Camera index (0, 1, ...)" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --quantize      Use INT8 quantization" << std::endl;
        std::cerr << "  --profile       Enable profiling" << std::endl;
        std::cerr << "  --conf <value>  Confidence threshold (default: 0.25)" << std::endl;
        std::cerr << "  --nms <value>   NMS threshold (default: 0.45)" << std::endl;
        std::cerr << "  --save <path>   Save output to file" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string input_source = argv[2];
    
    // Parse options
    bool use_quantization = false;
    bool enable_profiling = false;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    std::string output_path;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quantize") {
            use_quantization = true;
        } else if (arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--save" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }
    
    try {
        // Initialize logger
        Logger::instance().set_level(LogLevel::INFO);
        LOG_INFO("Deep Learning Inference Engine - YOLO Demo");
        
        // Enable profiling if requested
        if (enable_profiling) {
            Profiler::instance().enable(true);
        }
        
        // Load model
        LOG_INFO("Loading YOLO model from: %s", model_path.c_str());
        auto loader = ModelLoaderFactory::create_from_file(model_path);
        auto graph = loader->load(model_path);
        
        // Apply optimizations
        LOG_INFO("Applying optimizations...");
        
        // Layer fusion
        LayerFusionOptimizer fusion_opt;
        fusion_opt.optimize(*graph);
        
        // Quantization
        if (use_quantization) {
            LOG_INFO("Applying INT8 quantization...");
            QuantizationOptimizer quant_opt(8);
            quant_opt.optimize(*graph);
        }
        
        // Finalize graph
        graph->finalize();
        
        // Load class names
        auto class_names = load_coco_names("coco_classes.txt");
        
        // Create execution context
        ExecutionContext ctx;
        
        // Determine input source type
        cv::VideoCapture cap;
        cv::Mat frame;
        bool is_video = false;
        
        // Try to parse as camera index
        try {
            int camera_idx = std::stoi(input_source);
            cap.open(camera_idx);
            is_video = true;
        } catch (...) {
            // Not a camera index, try as file
            if (input_source.find(".mp4") != std::string::npos ||
                input_source.find(".avi") != std::string::npos ||
                input_source.find(".mov") != std::string::npos) {
                cap.open(input_source);
                is_video = true;
            } else {
                // Assume it's an image
                frame = cv::imread(input_source);
                if (frame.empty()) {
                    throw std::runtime_error("Failed to load input: " + input_source);
                }
            }
        }
        
        cv::VideoWriter writer;
        if (!output_path.empty() && is_video) {
            int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open(output_path, codec, 30.0, 
                       cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                               cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        }
        
        // Process frames
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (true) {
            // Get frame
            if (is_video) {
                cap >> frame;
                if (frame.empty()) break;
            }
            
            // Preprocess
            Tensor input = preprocess_yolo_image(frame);
            
            // Run inference
            auto inference_start = std::chrono::high_resolution_clock::now();
            auto outputs = graph->forward({input}, ctx);
            ctx.synchronize();
            auto inference_end = std::chrono::high_resolution_clock::now();
            
            // Process output
            auto detections = process_yolo_output(outputs[0], conf_threshold, nms_threshold);
            
            // Draw detections
            draw_detections(frame, detections, class_names);
            
            // Calculate FPS
            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            float fps = frame_count * 1000.0f / total_duration;
            
            auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                inference_end - inference_start).count() / 1000.0f;
            
            // Display info
            std::string info = "FPS: " + std::to_string(int(fps)) + 
                             " | Inference: " + std::to_string(inference_duration) + "ms" +
                             " | Objects: " + std::to_string(detections.size());
            
            cv::putText(frame, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Show or save result
            if (is_video) {
                cv::imshow("YOLO Detection", frame);
                if (writer.isOpened()) {
                    writer.write(frame);
                }
                
                // Check for exit
                if (cv::waitKey(1) == 27) break; // ESC key
            } else {
                // For single image
                cv::imshow("YOLO Detection", frame);
                if (!output_path.empty()) {
                    cv::imwrite(output_path, frame);
                    LOG_INFO("Saved output to: %s", output_path.c_str());
                }
                
                LOG_INFO("\nDetected %zu objects:", detections.size());
                for (const auto& det : detections) {
                    LOG_INFO("  %s: %.1f%% at [%.0f, %.0f, %.0f, %.0f]",
                            class_names[det.class_id].c_str(),
                            det.confidence * 100,
                            det.x - det.w/2, det.y - det.h/2,
                            det.x + det.w/2, det.y + det.h/2);
                }
                
                cv::waitKey(0);
                break;
            }
        }
        
        // Print summary statistics
        if (frame_count > 0) {
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            
            LOG_INFO("\nProcessing Summary:");
            LOG_INFO("  Total frames: %d", frame_count);
            LOG_INFO("  Total time: %.2f seconds", total_time / 1000.0f);
            LOG_INFO("  Average FPS: %.2f", frame_count * 1000.0f / total_time);
        }
        
        // Print profiling results
        if (enable_profiling) {
            LOG_INFO("\nProfiling Results:");
            Profiler::instance().print_summary();
            Profiler::instance().export_chrome_trace("yolo_trace.json");
        }
        
        // Cleanup
        if (cap.isOpened()) cap.release();
        if (writer.isOpened()) writer.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error: %s", e.what());
        return 1;
    }
    
    return 0;
}