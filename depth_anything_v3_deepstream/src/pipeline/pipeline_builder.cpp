#include "pipeline_builder.h"
#include <iostream>
#include <sstream>

std::string PipelineBuilder::build_source_branch() {
    std::stringstream ss;

    switch (config.pipeline.source_type) {
        case SourceType::CAMERA:
            ss << "v4l2src device=" << config.pipeline.source_uri << " ! "
               << "videoconvert ! "
               << "videorate ! "
               << "video/x-raw,format=RGBA,framerate=" << config.pipeline.framerate << "/1 ! "
               << "nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;

        case SourceType::FILE:
            ss << "nvurisrcbin uri=file://" << config.pipeline.source_uri
               << " disable-audio=true"
               << " ! mux.sink_0 ";
            break;

        case SourceType::IMAGE: {
            std::string uri = config.pipeline.source_uri;
            std::string ext;
            auto dot = uri.rfind('.');
            if (dot != std::string::npos) {
                ext = uri.substr(dot + 1);
                for (auto& c : ext) c = std::tolower(c);
            }
            std::string decoder;
            if      (ext == "jpg" || ext == "jpeg") decoder = "jpegdec";
            else if (ext == "png")                  decoder = "pngdec";
            else {
                std::cerr << "[ERROR] Unsupported image format '." << ext
                          << "'. Supported: jpg, jpeg, png\n";
                break;
            }
            ss << "filesrc location=" << uri << " ! "
               << decoder << " ! "
               << "imagefreeze ! "
               << "videoconvert ! "
               << "video/x-raw,format=RGBA,framerate=1/1 ! "
               << "nvvideoconvert ! "
               << "video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;
        }

        case SourceType::RTSP:
            ss << "nvurisrcbin uri=" << config.pipeline.source_uri
               << " latency=200"
               << " rtsp-reconnect-interval=30"
               << " rtsp-reconnect-attempts=-1"
               << " disable-audio=true"
               << " ! queue ! "
               << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;
        
        case SourceType::URI:
            ss << "nvurisrcbin uri=" << config.pipeline.source_uri
               << " disable-audio=true "
               << "! queue ! "
               << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
               << "queue ! mux.sink_0 ";
            break;
    }

    return ss.str();
}

std::string PipelineBuilder::build_muxer_config() {
    std::stringstream ss;
    ss << "nvstreammux name=mux "
       << "batch-size=" << config.pipeline.batch_size << " "
       << "width=" << config.pipeline.width << " "
       << "height=" << config.pipeline.height << " "
       << "live-source=" << config.pipeline.live_source << " "
       << "batched-push-timeout=" << config.pipeline.batched_push_timeout << " ! " 
       << "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! "
       << "nvdewarper config-file=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-dewarper-test/config_dewarper_fish_dist_correction.txt ! "
       << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! ";

    if (config.pipeline.show_original)
        ss << "tee name=t0 ";
        
    return ss.str();
}

// std::string PipelineBuilder::build_inference_and_sink() {
//     std::stringstream ss;
//     const std::string sync = config.pipeline.live_source ? "false" : "true";

//     // Sirf Original Stream ko screen par bhejein (No model, No tilemux)
//     ss << "fpsdisplaysink video-sink=nveglglessink text-overlay=false sync=" << sync;

//     return ss.str();
// }

std::string PipelineBuilder::build_inference_and_sink() {
    std::stringstream ss;
    const std::string sync = config.pipeline.live_source ? "false" : "true";

    if (!config.pipeline.show_original) {
        ss << "nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " interval=" << config.infer.interval << " ! "
           << "nveglglessink sync=" << sync;
    } else {
        const int W = config.pipeline.width;
        const int H = config.pipeline.height;
        ss << "nvstreammux name=tilemux "
           << "batch-size=2 "
           << "width=" << W << " "
           << "height=" << H << " "
           << "live-source=" << config.pipeline.live_source << " "
           << "batched-push-timeout=" << config.pipeline.batched_push_timeout << " ! "
           << "nvmultistreamtiler rows=1 columns=2 "
           << "width=" << (W * 2) << " height=" << H << " ! "
           << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
           << "nveglglessink sync=" << sync << " "
           // Original branch (Ab ye pehle se dewarped/undistorted frame lega)
           << "t0. ! queue leaky=downstream max-size-buffers=10 ! nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! tilemux.sink_0 "
           // Depth branch (Is model ko bhi undistorted frame milega)
           // << "t0. ! queue leaky=downstream max-size-buffers=10 ! nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " ! "
           << "t0. ! queue leaky=downstream max-size-buffers=10 ! nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " interval=" << config.infer.interval << " ! "
           << "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! tilemux.sink_1";
    }
    // } else {
    //     const int W = config.pipeline.width;
    //     const int H = config.pipeline.height;

    //     ss // Branch 1: Original View (Full 30 FPS, koi model delay nahi)
    //        << "t0. ! queue max-size-buffers=2 ! "
    //        << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
    //        << "nveglglessink name=sink_original sync=" << sync << " " 

    //        // Branch 2: Depth Anything Model View (Independent display window)
    //        << "t0. ! queue leaky=downstream max-size-buffers=1 ! "
    //        << "nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " ! "
    //        << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
    //        << "nveglglessink name=sink_depth sync=" << sync << " ";
    // }
    return ss.str();
}

// std::string PipelineBuilder::build_muxer_config() {
//     std::stringstream ss;
//     ss << "nvstreammux name=mux "
//        << "batch-size=" << config.pipeline.batch_size << " "
//        << "width=" << config.pipeline.width << " "
//        << "height=" << config.pipeline.height << " "
//        << "live-source=" << config.pipeline.live_source << " "
//        << "batched-push-timeout=" << config.pipeline.batched_push_timeout << " ! ";
//     if (config.pipeline.show_original)
//         ss << "tee name=t0 ";
//     return ss.str();
// }

// std::string PipelineBuilder::build_inference_and_sink() {
//     std::stringstream ss;
//     const std::string sync = config.pipeline.live_source ? "false" : "true";

//     if (!config.pipeline.show_original) {
//         ss << "nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " ! "
//            << "nveglglessink sync=" << sync;
//     } else {
//         // nvvideoconvert before tilemux strips DeepStream batch metadata so
//         // nvstreammux can accept the frames as plain NVMM sources.
//         const int W = config.pipeline.width;
//         const int H = config.pipeline.height;
//         ss << "nvstreammux name=tilemux "
//            << "batch-size=2 "
//            << "width=" << W << " "
//            << "height=" << H << " "
//            << "live-source=" << config.pipeline.live_source << " "
//            << "batched-push-timeout=4000 ! "
//            << "nvmultistreamtiler rows=1 columns=2 "
//            << "width=" << (W * 2) << " height=" << H << " ! "
//            << "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
//            << "nveglglessink sync=" << sync << " "
//            // Original branch → left tile
//            << "t0. ! queue ! nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! tilemux.sink_0 "
//            // Depth branch → right tile
//            << "t0. ! nvinfer name=depth_anything config-file-path=" << config.infer.config_path << " ! "
//            << "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! tilemux.sink_1";
//     }
//     return ss.str();
// }


std::string PipelineBuilder::build_pipeline_description() {
    std::stringstream ss;
    ss << build_source_branch()
       << build_muxer_config()
       << build_inference_and_sink();
    return ss.str();
}

GstElement* PipelineBuilder::create_pipeline(GError** error) {
    std::string pipeline_desc = build_pipeline_description();
    return gst_parse_launch(pipeline_desc.c_str(), error);
}

bool PipelineProbeAttacher::attach_probe_to_element(
    GstElement* pipeline,
    const char* element_name,
    GstPadProbeCallback callback,
    gpointer user_data,
    GDestroyNotify destroy_notify,
    bool attach_to_src)
{
    GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), element_name);
    if (!element) {
        std::cerr << "Could not find element named '" << element_name << "'\n";
        return false;
    }

    const char* pad_name = attach_to_src ? "src" : "sink";
    GstPad* pad = gst_element_get_static_pad(element, pad_name);
    if (!pad) {
        std::cerr << "Could not get " << pad_name << " pad of '" << element_name << "'\n";
        gst_object_unref(element);
        return false;
    }

    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, callback, user_data, destroy_notify);
    gst_object_unref(pad);
    gst_object_unref(element);

    return true;
}
