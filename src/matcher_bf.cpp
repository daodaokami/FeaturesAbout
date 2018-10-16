//
// Created by lut on 18-10-12.
//

#include "../include/matcher_bf.h"
#include "matcher.h"

namespace suo15features{
    Matcher_bf::Matcher_bf(Options opts){
        //先设置匹配matcher的参数

    }

    vector<CP> Matcher_bf::GetMatchedKeypoints(const vector<cv::KeyPoint> &keypoints_1,
                                                             const cv::Mat &descriptors_1,
                                                             const vector<cv::KeyPoint> &keypoints_2,
                                                             const cv::Mat &descriptors_2){
        vector<CP> keypointsCP;


        return keypointsCP;
    };

}