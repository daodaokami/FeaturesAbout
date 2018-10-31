//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_BF_H
#define LUT15VO_MATCHER_BF_H

#include "matcher.h"

namespace suo15features {

    class Matcher_bf:public Matcher{
    private:
        Matcher_options options;
    public:
        Matcher_bf(){}
        Matcher_bf(const Matcher_options& opts);
        virtual vector<CP> GetMatchedKeypoints(const vector<cv::KeyPoint> &keypoints_1,
                                                                 const cv::Mat &descriptors_1,
                                                                 const vector<cv::KeyPoint> &keypoints_2,
                                                                 const cv::Mat &descriptors_2);
    };
}

#endif //LUT15VO_BF_H
