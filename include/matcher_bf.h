//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_BF_H
#define LUT15VO_MATCHER_BF_H

#include "common_include.h"
#include "matcher.h"

namespace suo15features {
    struct Options{
        int descriptor_length;
        float lowe_ratio_threshold;
        float distance_threshold;
    };
    class Matcher_bf:public Matcher{
    private:
        Options _config;
    public:
        Matcher_bf(){}
        Matcher_bf(Options opts);
        virtual vector<CP> GetMatchedKeypoints(const vector<cv::KeyPoint> &keypoints_1,
                                                                 const cv::Mat &descriptors_1,
                                                                 const vector<cv::KeyPoint> &keypoints_2,
                                                                 const cv::Mat &descriptors_2);
    };
}

#endif //LUT15VO_BF_H
