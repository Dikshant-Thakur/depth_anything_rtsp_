#pragma once

#include "gst_headers.h"
#include <cuda_runtime_api.h>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <vector>

static inline size_t elem_size_from_infer_dtype(NvDsInferDataType t) {
    switch (t) {
      case NvDsInferDataType::FLOAT: return 4;
      case NvDsInferDataType::HALF:  return 2;
      case NvDsInferDataType::INT8:  return 1;
      case NvDsInferDataType::INT32: return 4;
      default: return 0;
    }
}

static inline bool is_nv12_color_format(NvBufSurfaceColorFormat fmt) {
    return fmt == NVBUF_COLOR_FORMAT_NV12 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_2020;
}

// Find the NvDsInferTensorMeta for a given unique_id in frame user meta.
// DA V3 is a primary nvinfer — output tensor is in frame_meta->frame_user_meta_list.
static NvDsInferTensorMeta*
find_tensor_meta_for_uid(NvDsFrameMeta* frame_meta, guint64 uid)
{
    for (NvDsMetaList* l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta* um = (NvDsUserMeta*)l->data;
        if (!um) continue;
        if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;

        auto* tmeta = (NvDsInferTensorMeta*)um->user_meta_data;
        if (tmeta && tmeta->unique_id == uid) return tmeta;
    }
    return nullptr;
}

static int find_layer_index(NvDsInferTensorMeta* tmeta, const std::string& name) {
    for (unsigned i = 0; i < tmeta->num_output_layers; ++i) {
        const char* ln = tmeta->output_layers_info[i].layerName;
        if (ln && name == ln) return (int)i;
    }
    return -1;
}

// Copy NvBufSurface to ensure write independence (needed when buffer is mapped read-only)
static inline NvBufSurface* copy_and_replace_buffer_surface(
    GstBuffer* buf,
    GstMapInfo& in_out_map,
    NvBufSurface* surface)
{
    if (!buf || !surface) return nullptr;

    if (gst_buffer_n_memory(buf) == 1) {
        GstMemory* mem = gst_buffer_peek_memory(buf, 0);
        if (mem && gst_memory_is_writable(mem)) {
            return surface;
        }
    }

    NvBufSurface* new_surface = nullptr;
    NvBufSurfaceCreateParams create_params{};
    create_params.gpuId = surface->gpuId;
    create_params.width = surface->surfaceList[0].width;
    create_params.height = surface->surfaceList[0].height;
    create_params.size = 0;
    create_params.colorFormat = surface->surfaceList[0].colorFormat;
    create_params.layout = surface->surfaceList[0].layout;
    create_params.memType = surface->memType;

    if (NvBufSurfaceCreate(&new_surface, surface->batchSize, &create_params) != 0) {
        return nullptr;
    }

    if (NvBufSurfaceCopy(surface, new_surface) != 0) {
        NvBufSurfaceDestroy(new_surface);
        return nullptr;
    }

    gst_buffer_unmap(buf, &in_out_map);
    gst_buffer_remove_all_memory(buf);

    GstMemory* mem = gst_memory_new_wrapped(
        GST_MEMORY_FLAG_NO_SHARE,
        new_surface,
        sizeof(NvBufSurface),
        0,
        sizeof(NvBufSurface),
        new_surface,
        [](gpointer data) { NvBufSurfaceDestroy((NvBufSurface*)data); });

    gst_buffer_append_memory(buf, mem);

    if (!gst_buffer_map(buf, &in_out_map, GST_MAP_READWRITE)) {
        return nullptr;
    }

    return (NvBufSurface*)in_out_map.data;
}
