//
// Created by lut on 18-10-31.
//

#ifndef LUT15VO_VISUALIZE_H
#define LUT15VO_VISUALIZE_H

#include "common_include.h"

namespace suo15features{
    void ShowImgWithKeypoints(const string winname, const cv::Mat& img, const vector<cv::Point2f>& positions);

    void ShowImgWithMatches(const string winname, const cv::Mat& img,
                            const vector<cv::Point2f>& positions_1,
                            const vector<cv::Point2f>& positions_2,
                            const Result& result);

};


#endif //LUT15VO_VISUALIZE_H
