#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "fisheye_undistort.h"
#include <opencv2/opencv.hpp>

// Apne publisher ka header include karein
// #include "my_image_publisher.h"

static FisheyeUndistorter* g_undistorter = nullptr;

static GstPadProbeReturn raw_undistort_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = reinterpret_cast<GstBuffer*>(info->data);
    if (!buf) return GST_PAD_PROBE_OK;

    GstMapInfo in_map;
    if (!gst_buffer_map(buf, &in_map, GST_MAP_READ)) return GST_PAD_PROBE_OK;

    NvBufSurface *surface = reinterpret_cast<NvBufSurface*>(in_map.data);
    if (!surface || surface->numFilled == 0) {
        gst_buffer_unmap(buf, &in_map);
        return GST_PAD_PROBE_OK;
    }

    // Map surface to CPU for OpenCV access
    NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ);
    NvBufSurfaceSyncForCpu(surface, -1, -1);

    NvBufSurfaceParams &params = surface->surfaceList[0];
    int width = params.width;
    int height = params.height;

    if (!g_undistorter) {
        g_undistorter = new FisheyeUndistorter(width, height);
    }

    // Wrap NV12/RGBA buffer into cv::Mat
    // Note: Agar pixel format NVMM NV12 hai:
    cv::Mat nv12_mat(height + height / 2, width, CV_8UC1, params.mappedAddr.addr[0], params.pitch);
    cv::Mat bgr_mat;
    cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);

    // Undistort execution
    cv::Mat undistorted_raw = g_undistorter->process(bgr_mat);

    // -------------------------------------------------------------
    // Yahan apni publisher file ka publish call daalein:
    // my_publisher->publish_raw_image(undistorted_raw);
    // -------------------------------------------------------------

    NvBufSurfaceUnMap(surface, -1, -1);
    gst_buffer_unmap(buf, &in_map);

    return GST_PAD_PROBE_OK;
}
