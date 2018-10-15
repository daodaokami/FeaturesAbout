//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESCRIPTOR_H
#define LUT15VO_DESCRIPTOR_H

#include "common_include.h"

namespace suo15features {
    class Descriptor {
        //因为能够派生出各种类型的描述子
                //orb256
                //sift128
                //surf64
        //那么descriptor主要提供的是一个通用的接口，能够提供方便的子类的描述子提取
    public:
        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, const vector<cv::KeyPoint>& keypoints);
    };
}

#endif //LUT15VO_DESCRIPTOR_H
