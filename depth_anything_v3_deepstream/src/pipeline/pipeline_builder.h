#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"
#include <string>

class PipelineBuilder {
public:
    PipelineBuilder(const AppConfig& cfg) : config(cfg) {}

    std::string build_pipeline_description();
    GstElement* create_pipeline(GError** error);

private:
    const AppConfig& config;

    std::string build_source_branch();
    std::string build_muxer_config();
    std::string build_inference_and_sink();
};

class PipelineProbeAttacher {
public:
    static bool attach_probe_to_element(
        GstElement* pipeline,
        const char* element_name,
        GstPadProbeCallback callback,
        gpointer user_data,
        GDestroyNotify destroy_notify,
        bool attach_to_src = true);
};
