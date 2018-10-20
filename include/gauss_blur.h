//
// Created by lut on 18-10-19.
//
#ifndef LUT15VO_GAUSS_BLUR_H
#define LUT15VO_GAUSS_BLUR_H

#include "common_include.h"
#include "accum_conv.h"
namespace suo15features{
    class Gauss_Blur {
    public:
        cv::Mat rescale_double_size_supersample(const cv::Mat &img);
        cv::Mat rescale_half_size_gaussian(const cv::Mat &img, float sigma= 0.866025403784439f);
    };
}

#endif //LUT15VO_GAUSS_BLUR_H
