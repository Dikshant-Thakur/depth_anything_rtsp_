#pragma once

#include <string>
#include <cstdint>

enum class SourceType {
    CAMERA,  // USB/V4L2 camera (e.g., /dev/video0)
    FILE,    // Video file (mp4, avi, etc.)
    IMAGE,   // Single image (jpg, png, etc.)
    RTSP,    // RTSP stream
    URI      // Generic URI (http, file://, etc.)
};

struct PipelineConfig {
    SourceType source_type = SourceType::URI;
    std::string source_uri = "rtsp://admin:Video_2026@192.168.34.135:554/rtsp/streaming?channel=01&subtype=0";
    int framerate = 30;
    int batch_size = 1;
    int width = 1280;
    int height = 720;
    int live_source = 1;
    int batched_push_timeout = 33333;
    bool show_original = false;
};

// All nvinfer-related settings for the single depth model in the pipeline.
// the TRT engine is built on first run and cached at model-engine-file in config_path.
struct InferConfig {
    std::string config_path   = "/home/moliris-ros2/depth_anything_v3_deepstream/depth_anything_v3_deepstream/configs/config_infer_depth_anything_v3.txt";
    uint32_t    gie_uid       = 1;        // must match gie-unique-id in config_path
    std::string output_layer  = "depth"; 
    int interval = 2;   
};

struct DebugConfig {
    bool enabled = false;
    uint64_t initial_frames = 10;
    uint64_t periodic_interval = 120;
    std::string dot_file_path = "/depth_anything_v3_deepstream/depth_anything_v3_deepstream/build/pipeline.dot";
};

// Master configuration
struct AppConfig {
    PipelineConfig pipeline;
    InferConfig    infer;
    DebugConfig    debug;


    static AppConfig create_default() {
        return AppConfig{};
    }
};
