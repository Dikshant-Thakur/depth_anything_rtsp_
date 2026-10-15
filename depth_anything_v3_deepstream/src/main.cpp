#include "utils/gst_headers.h"
#include "config/app_config.h"
#include "pipeline/pipeline_builder.h"
#include "probes/depth_probe.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>

int main(int argc, char *argv[]) {
    AppConfig app_config = AppConfig::create_default();

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--source-type" && i + 1 < argc) {
            std::string type = argv[++i];
            if      (type == "camera") app_config.pipeline.source_type = SourceType::CAMERA;
            else if (type == "file")   { app_config.pipeline.source_type = SourceType::FILE;  app_config.pipeline.live_source = 0; }
            else if (type == "image")  { app_config.pipeline.source_type = SourceType::IMAGE; app_config.pipeline.live_source = 0; }
            else if (type == "rtsp")   app_config.pipeline.source_type = SourceType::RTSP;
            else if (type == "uri")    { app_config.pipeline.source_type = SourceType::URI;   app_config.pipeline.live_source = 0; }
            else std::cerr << "Unknown source type: " << type << " (use: camera, file, image, rtsp, uri)\n";
        }
        else if (a == "--source-uri" && i + 1 < argc) {
            app_config.pipeline.source_uri = argv[++i];
        }
        else if (a == "--framerate" && i + 1 < argc) {
            app_config.pipeline.framerate = std::stoi(argv[++i]);
        }
        else if (a == "--config" && i + 1 < argc) {
            app_config.infer.config_path = argv[++i];
        }
        else if (a == "--debug" && i + 1 < argc) {
            std::string val = argv[++i];
            app_config.debug.enabled = (val == "true" || val == "1");
        }
        else if (a == "--dot-file" && i + 1 < argc) {
            app_config.debug.dot_file_path = argv[++i];
        }
        else if (a == "--show-original" && i + 1 < argc) {
            std::string val = argv[++i];
            app_config.pipeline.show_original = (val == "true" || val == "1");
        }
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Options:\n"
                      << "  --source-type TYPE        Source type: camera, file, image, rtsp, uri (default: camera)\n"
                      << "  --source-uri URI           Source URI (device path, file path, or stream URL)\n"
                      << "  --framerate FPS            Frame rate (default: 30)\n"
                      << "  --config PATH              DeepStream config file path\n"
                      << "  --show-original [true|false]  Show original frame alongside depth (default: false)\n"
                      << "  --debug [true|false]       Enable debug mode (default: false)\n"
                      << "  --dot-file PATH            Path for pipeline DOT file\n"
                      << "  -h, --help                 Show this help message\n";
            return 0;
        }
    }

    // Convert relative file paths to absolute
    if (app_config.pipeline.source_type == SourceType::FILE) {
        namespace fs = std::filesystem;
        fs::path file_path(app_config.pipeline.source_uri);
        if (file_path.is_relative()) {
            app_config.pipeline.source_uri = fs::absolute(file_path).string();
        }
    }

    // Verify model config exists
    {
        std::ifstream f(app_config.infer.config_path);
        if (!f) {
            std::cerr << "[ERROR] Model config not found: " << app_config.infer.config_path << "\n";
            std::cerr << "[ERROR] Run scripts/export_onnx.py first, then update the config path.\n";
            return 1;
        }
    }

    if (app_config.debug.enabled) {
        std::string full_path = app_config.debug.dot_file_path;
        size_t last_slash = full_path.find_last_of("/\\");
        std::string dot_dir = (last_slash != std::string::npos) ? full_path.substr(0, last_slash) : ".";
        g_setenv("GST_DEBUG_DUMP_DOT_DIR", dot_dir.c_str(), TRUE);
        std::cout << "[DEBUG] DOT dir: " << dot_dir << "\n";
    }

    gst_init(&argc, &argv);

    PipelineBuilder builder(app_config);

    if (app_config.debug.enabled) {
        std::cout << "[DEBUG] Pipeline:\n" << builder.build_pipeline_description() << "\n\n";
    }

    GError* error = nullptr;
    GstElement* pipeline = builder.create_pipeline(&error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline.\n";
        if (error) std::cerr << "Error: " << error->message << "\n";
        return 1;
    }

    // Attach depth probe to DA V3 nvinfer src pad
    auto* depth_handler = new DepthProbeHandler(app_config);
    if (!PipelineProbeAttacher::attach_probe_to_element(
            pipeline, "depth_anything",
            depth_src_pad_probe_wrapper,
            depth_handler,
            [](gpointer data) { delete reinterpret_cast<DepthProbeHandler*>(data); })) {
        gst_object_unref(pipeline);
        return 1;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    if (app_config.debug.enabled) {
        std::string full_path = app_config.debug.dot_file_path;
        size_t last_slash = full_path.find_last_of("/\\");
        std::string dot_dir = (last_slash != std::string::npos) ? full_path.substr(0, last_slash) : ".";
        std::string filename = (last_slash != std::string::npos) ? full_path.substr(last_slash + 1) : full_path;
        std::string dot_name = filename;
        if (dot_name.size() > 4 && dot_name.substr(dot_name.size() - 4) == ".dot") {
            dot_name = dot_name.substr(0, dot_name.size() - 4);
        }
        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, dot_name.c_str());
        std::cout << "[DEBUG] DOT file: " << dot_dir << "/" << dot_name << ".dot\n";
        std::cout << "[DEBUG] Convert: dot -Tpng " << dot_dir << "/" << dot_name << ".dot -o pipeline.png\n";
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    bool running = true;
    while (running) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, GST_CLOCK_TIME_NONE,
            (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        if (!msg) continue;

        switch (GST_MESSAGE_TYPE(msg)) {
          case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::cerr << "GStreamer ERROR: " << (err ? err->message : "unknown") << "\n";
            if (dbg) std::cerr << "Debug: " << dbg << "\n";
            if (err) g_error_free(err);
            if (dbg) g_free(dbg);
            running = false;
            break;
          }
          case GST_MESSAGE_EOS:
            running = false;
            break;
          default:
            break;
        }
        gst_message_unref(msg);
    }

    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}
