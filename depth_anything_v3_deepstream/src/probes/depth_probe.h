#pragma once

#include "utils/gst_headers.h"
#include "config/app_config.h"
#include "utils_cuda/depth_percentile.h"

class DepthProbeHandler {
public:
    DepthProbeHandler(const AppConfig& cfg)
        : config(cfg), call_idx(0) {}

    ~DepthProbeHandler(); 

    GstPadProbeReturn handle_buffer(GstPad* pad, GstPadProbeInfo* info);

private:
    const AppConfig& config;
    uint64_t call_idx;

    bool should_debug() const;
    bool ensure_buffer_writable(GstPadProbeInfo* info);
    NvBufSurface* map_buffer(GstBuffer* buf, GstMapInfo& map_info);
    bool validate_surface(NvBufSurface* surface, bool debug);


    void* cached_y_dev = nullptr;
    void* cached_uv_dev = nullptr;
    int cached_w = 0, cached_h = 0;
    int cached_pitchY = 0, cached_pitchUV = 0;
    bool has_cached_frame = false;

    struct DepthTensorInfo {
        const float* depth_dev;
        int width;
        int height;
    };

    // Extract depth tensor from NvDsInferTensorMeta in frame_user_meta_list
    bool extract_depth_tensor(
        NvDsFrameMeta* frame_meta,
        DepthTensorInfo& out_info,
        bool debug);

    bool process_frame_depth(
        NvBufSurface* surface,
        NvDsFrameMeta* frame_meta,
        const DepthTensorInfo& depth_info,
        bool debug);
};

GstPadProbeReturn depth_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data);
