//
// Created by lut on 18-10-18.
//

#ifndef LUT15VO_SIFT_KEYPOINT_H
#define LUT15VO_SIFT_KEYPOINT_H

#include "common_include.h"
namespace suo15features{
    class Sift_KeyPoint: public cv::KeyPoint{
    public:
        int sample;
        //可以直接访问kp
        Sift_KeyPoint();
        Sift_KeyPoint(const cv::KeyPoint& kp);
    };

}
#endif //LUT15VO_SIFT_KEYPOINT_H
