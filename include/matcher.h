//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_H
#define LUT15VO_MATCHER_H

#include "common_include.h"

namespace suo15features {
    typedef pair<size_t, size_t> CP;
    class Matcher {
    public:
        virtual vector<CP> GetMatchedKeypoints(const vector<cv::KeyPoint>& keypoints_1, const cv::Mat& descriptors_1,
                const vector<cv::KeyPoint>& keypoints_2, const cv::Mat& descriptors_2);
    };
}


#endif //LUT15VO_MATCHER_H
