//
// Created by lut on 18-10-18.
//
#include "sift_keypoint.h"
namespace suo15features{
    Sift_KeyPoint::Sift_KeyPoint(){}

    Sift_KeyPoint::Sift_KeyPoint(const cv::KeyPoint& kp):cv::KeyPoint(kp){
        this->sample = -1;
        this->scale = -1;
    }
}