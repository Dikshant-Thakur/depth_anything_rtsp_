#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

class FisheyeUndistorter {
public:
    FisheyeUndistorter(int width, int height) {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 
            500.8600, 0.0,      642.8416,
            0.0,      508.5673, 321.0878,
            0.0,      0.0,      1.0);

        cv::Mat D = (cv::Mat_<double>(4, 1) << 
            0.02209469, 
            0.11496114, 
            -0.20511922, 
            0.08565259);

        cv::Mat newK;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            K, D, cv::Size(width, height), cv::Mat::eye(3, 3, CV_64F), newK, 1.0);

        cv::fisheye::initUndistortRectifyMap(
            K, D, cv::Mat(), newK, cv::Size(width, height), CV_16SC2, map1_, map2_);
    }

    cv::Mat process(const cv::Mat& input_bgr) {
        cv::Mat undistorted;
        cv::remap(input_bgr, undistorted, map1_, map2_, cv::INTER_LINEAR);
        return undistorted;
    }

private:
    cv::Mat map1_, map2_;
};
