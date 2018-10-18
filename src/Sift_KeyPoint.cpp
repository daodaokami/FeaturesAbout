//
// Created by lut on 18-10-18.
//
#include "Sift_KeyPoint.h"
namespace suo15features{
    Sift_KeyPoint::Sift_KeyPoint(){}

    Sift_KeyPoint::Sift_KeyPoint(const cv::KeyPoint& kp):cv::KeyPoint(kp){
        sample = -1;
    }
}