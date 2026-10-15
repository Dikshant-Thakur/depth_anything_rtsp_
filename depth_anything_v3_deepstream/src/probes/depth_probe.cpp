#include "depth_probe.h"
#include "utils/gst_utils.h"
#include <iostream>
#include "fisheye_undistort.h"

static FisheyeUndistorter* g_undistorter = nullptr;

bool DepthProbeHandler::should_debug() const {
    return config.debug.enabled &&
           ((call_idx <= config.debug.initial_frames) ||
            (call_idx % config.debug.periodic_interval == 0));
}

bool DepthProbeHandler::ensure_buffer_writable(GstPadProbeInfo* info) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return false;

    if (!gst_buffer_is_writable(buf)) {
        GstBuffer* wbuf = gst_buffer_make_writable(buf);
        if (!wbuf) return false;
        GST_PAD_PROBE_INFO_DATA(info) = wbuf;
    }
    return true;
}

NvBufSurface* DepthProbeHandler::map_buffer(GstBuffer* buf, GstMapInfo& map_info) {
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        return nullptr;
    }
    return (NvBufSurface*)map_info.data;
}

bool DepthProbeHandler::validate_surface(NvBufSurface* surface, bool debug) {
    if (!surface) return false;

    if (debug) {
        std::cout << "\n[DEPTH/CUDA] call #" << call_idx
                  << " memType=" << (int)surface->memType
                  << " numFilled=" << surface->numFilled
                  << " batchSize=" << surface->batchSize
                  << "\n";
    }

    return surface->memType == NVBUF_MEM_CUDA_DEVICE;
}

// Extract depth tensor from NvDsInferTensorMeta.
bool DepthProbeHandler::extract_depth_tensor(
    NvDsFrameMeta* frame_meta,
    DepthTensorInfo& out_info,
    bool debug)
{
    NvDsInferTensorMeta* tmeta = find_tensor_meta_for_uid(
        frame_meta, config.infer.gie_uid);

    if (!tmeta) {
        if (debug) {
            std::cout << "[DEPTH] frame=" << frame_meta->frame_num
                      << " no NvDsInferTensorMeta for uid="
                      << config.infer.gie_uid << "\n";
        }
        return false;
    }

    int li = find_layer_index(tmeta, config.infer.output_layer);
    if (li < 0) {
        if (debug) {
            std::cout << "[DEPTH] frame=" << frame_meta->frame_num
                      << " layer '" << config.infer.output_layer << "' not found\n";
            std::cout << "[DEPTH] Available layers: ";
            for (unsigned i = 0; i < tmeta->num_output_layers; ++i) {
                std::cout << "'" << (tmeta->output_layers_info[i].layerName ? tmeta->output_layers_info[i].layerName : "null") << "' ";
            }
            std::cout << "\n";
        }
        return false;
    }

    NvDsInferLayerInfo& layer = tmeta->output_layers_info[li];

    if (layer.dataType != NvDsInferDataType::FLOAT) {
        if (debug) {
            std::cout << "[DEPTH] frame=" << frame_meta->frame_num
                      << " depth dtype not FLOAT\n";
        }
        return false;
    }

    void* depth_dev_void = (tmeta->out_buf_ptrs_dev ?
                            tmeta->out_buf_ptrs_dev[li] : nullptr);
    if (!depth_dev_void) {
        if (debug) {
            std::cout << "[DEPTH] frame=" << frame_meta->frame_num
                      << " no out_buf_ptrs_dev\n";
        }
        return false;
    }

    int H = 0, W = 0;
    if (layer.inferDims.numDims >= 2) {
        H = layer.inferDims.d[layer.inferDims.numDims - 2];
        W = layer.inferDims.d[layer.inferDims.numDims - 1];
    }

    if (H <= 0 || W <= 0) return false;

    // For visualization: percentile normalization is applied in process_frame_depth.
    out_info.depth_dev = (const float*)depth_dev_void;
    out_info.width = W;
    out_info.height = H;
    return true;
}

bool DepthProbeHandler::process_frame_depth(
    NvBufSurface* surface,
    NvDsFrameMeta* frame_meta,
    const DepthTensorInfo& depth_info,
    bool debug)
{
    const int b = (int)frame_meta->batch_id;
    if (b < 0 || b >= (int)surface->batchSize) return false;

    NvBufSurfaceParams& sl = surface->surfaceList[b];

    if (!is_nv12_color_format(sl.colorFormat)) {
        if (debug) {
            std::cout << "[DEPTH] surface not NV12 (fmt=" << (int)sl.colorFormat << ")\n";
        }
        return false;
    }

    uint8_t* base = (uint8_t*)sl.dataPtr;
    int pitchY = (int)sl.pitch;
    int pitchUV = pitchY;
    int outW = (int)sl.width;
    int outH = (int)sl.height;

    uint8_t* y_dev = base;
    uint8_t* uv_dev = base + (size_t)pitchY * (size_t)outH;

    cudaStream_t stream = 0;
    cudaError_t e = depth_to_nv12_colormap_percentile_launch(
        depth_info.depth_dev, depth_info.width, depth_info.height,
        y_dev, uv_dev,
        outW, outH, pitchY, pitchUV,
        stream);

    if (e != cudaSuccess) {
        if (debug) std::cerr << "[DEPTH] CUDA launch error: " << cudaGetErrorString(e) << "\n";
        return false;
    }

    cudaError_t e2 = cudaStreamSynchronize(stream);
    if (e2 != cudaSuccess) {
        if (debug) std::cerr << "[DEPTH] CUDA sync error: " << cudaGetErrorString(e2) << "\n";
        return false;
    }



    // ================== VISUAL CENTER MARKER (crosshair) ==================

    // NV12 format - image has 2 types of data. 
    // Y Plane (Brightness / Luma) - Memory ka pehla hissa, only black and white.
    // UV Plane (Color / Chroma) - Memory ka dusra hissa, color information.
    // ====== IMPORTANT - In GPU memory layout, Y plane is first, followed by UV plane ======
    // ====== And the order is not 2D, its 1D ======
    {
        // Depth map ka center (W/2, H/2) proportionally output surface pe map hota hai
        int marker_x = outW / 2;
        int marker_y = outH / 2;

        int cross_size = 8;   // line ki lambai (pixels), zaroorat ho toh badhao/ghatao
        int thickness = 1;     // line ki motai

        // Horizontal line (Y plane pe, white = 255)
        int h_x = std::max(0, marker_x - cross_size);
        int h_w = std::min(cross_size * 2, outW - h_x);
        if (h_w > 0) {
            cudaMemset2D(
                y_dev + (size_t)(marker_y - thickness/2) * pitchY + h_x,
                pitchY, 255, h_w, thickness);
        }

        // Vertical line (Y plane pe, white = 255)
        // y_dev - pointer, top left corner of Y plane. First pixel of Y plane.
        int v_y = std::max(0, marker_y - cross_size);
        int v_h = std::min(cross_size * 2, outH - v_y);
        if (v_h > 0) {
            cudaMemset2D(
                y_dev + (size_t)v_y * pitchY + (marker_x - thickness/2),
                pitchY, 255, thickness, v_h);
        } //

        // Chroma neutral rakho taaki marker white/gray dikhe, color-shift na ho
        int uv_marker_y = marker_y / 2;
        cudaMemset2D(
            uv_dev + (size_t)(uv_marker_y - thickness/2) * pitchUV + (h_x),
            pitchUV, 128, h_w, thickness);
    }
    // =========================================================================

    // ---- NEW: cache the successfully rendered NV12 depth colormap ----
    if (!has_cached_frame || cached_w != outW || cached_h != outH) {
        if (cached_y_dev) cudaFree(cached_y_dev);
        if (cached_uv_dev) cudaFree(cached_uv_dev);
        cudaMalloc(&cached_y_dev, (size_t)pitchY * outH);
        cudaMalloc(&cached_uv_dev, (size_t)pitchUV * (outH / 2));
        cached_w = outW; cached_h = outH;
        cached_pitchY = pitchY; cached_pitchUV = pitchUV;
    }
    cudaMemcpyAsync(cached_y_dev, y_dev, (size_t)pitchY * outH, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cached_uv_dev, uv_dev, (size_t)pitchUV * (outH / 2), cudaMemcpyDeviceToDevice, stream);    cudaStreamSynchronize(stream);
    has_cached_frame = true;
    // --------------------------------------------------------------------

    // ================== METRIC DEPTH CALCULATION  ==================
    int W = depth_info.width;
    int H = depth_info.height;
    int center_x = W / 2;
    int center_y = H / 2;
    int center_idx = center_y * W + center_x;

    float raw_center_val = 0.0f;
    // GPU pointer se center pixel value ko CPU host memory par copy karna:
    if (frame_meta->frame_num % 10 == 0) {
        cudaMemcpy(&raw_center_val, depth_info.depth_dev + center_idx, sizeof(float), cudaMemcpyDeviceToHost);
        // Focal pixels calculation: (fx + fy) / 2
        // Raw camera intrinsics: fx = 500.8600, fy = 508.5673
        const float fx_orig = 246.83594407f;
        const float fy_orig = 233.3260204f;
        // const float orig_w = 1920.0f;
        // const float orig_h = 1080.0f;
        // const float model_w = (float)depth_info.width;
        // const float model_h = (float)depth_info.height;
        float focal_pixels = (fx_orig + fy_orig) / 2.0f;

        // DA3METRIC-LARGE Formula:
        float metric_depth_m = (focal_pixels * raw_center_val) / 300.0f;

        std::cout << "[METRIC DEPTH] Frame " << frame_meta->frame_num
                << " | Center (" << center_x << ", " << center_y << ")"
                << " | Raw Val: " << raw_center_val
                << " | Width:" << W
                << " | Height:" << H
                << " | Distance: " << metric_depth_m << " meters\n";
        // =================== METRIC DEPTH CALCULATION (END) ===================
    }


    if (debug) {
        std::cout << "[DEPTH] frame=" << frame_meta->frame_num
                  << " depth " << depth_info.width << "x" << depth_info.height
                  << " Depth Info " << depth_info.depth_dev
                  << " -> NV12 " << outW << "x" << outH << "\n";
    }

    return true;
}

GstPadProbeReturn DepthProbeHandler::handle_buffer(
    GstPad* pad,
    GstPadProbeInfo* info)
{
    (void)pad;

    call_idx++;
    const bool debug = should_debug();

    if (!ensure_buffer_writable(info)) return GST_PAD_PROBE_OK;

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    GstMapInfo in_map{};
    NvBufSurface* surface = map_buffer(buf, in_map);
    if (!surface) return GST_PAD_PROBE_DROP;

    if (!validate_surface(surface, debug)) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_DROP;
    }

    surface = copy_and_replace_buffer_surface(buf, in_map, surface);
    if (!surface) {
        if (debug) std::cerr << "[DEPTH] Failed to copy surface\n";
        return GST_PAD_PROBE_DROP;
    }

    int updated = 0;
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta* fmeta = (NvDsFrameMeta*)l_frame->data;
        if (!fmeta) continue;

        DepthTensorInfo depth_info;
        if (extract_depth_tensor(fmeta, depth_info, debug)) {
            // Naya inference output mila -> process + cache update
            if (process_frame_depth(surface, fmeta, depth_info, debug)) updated++;
        } else if (has_cached_frame) {
            // Skip hua frame (interval) -> cached depth reuse karo
            const int b = (int)fmeta->batch_id;
            if (b >= 0 && b < (int)surface->batchSize) {
                NvBufSurfaceParams& sl = surface->surfaceList[b];

                if (sl.pitch == cached_pitchY && sl.height == cached_h && sl.width == cached_w) {
                    uint8_t* base = (uint8_t*)sl.dataPtr;
                    uint8_t* y_dev = base;
                    uint8_t* uv_dev = base + (size_t)sl.pitch * (size_t)sl.height;

                    cudaMemcpy(y_dev, cached_y_dev, (size_t)cached_pitchY * cached_h, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(uv_dev, cached_uv_dev, (size_t)cached_pitchUV * (cached_h / 2), cudaMemcpyDeviceToDevice);
                    updated++;
                } else {
                    std::cerr << "[DEPTH] Size mismatch! Cannot restore cache.\n";
                }
            }
        }


        // if (process_frame_depth(surface, fmeta, depth_info, debug)) updated++;
    }

    if (debug && updated == 0) {
        std::cout << "[DEPTH] no frames updated\n";
    }

    gst_buffer_unmap(buf, &in_map);
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn depth_src_pad_probe_wrapper(
    GstPad* pad,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* handler = reinterpret_cast<DepthProbeHandler*>(user_data);
    return handler->handle_buffer(pad, info);
}

DepthProbeHandler::~DepthProbeHandler() {
    if (cached_y_dev) cudaFree(cached_y_dev);
    if (cached_uv_dev) cudaFree(cached_uv_dev);
}