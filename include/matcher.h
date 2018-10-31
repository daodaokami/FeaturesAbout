//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_H
#define LUT15VO_MATCHER_H

#include "common_include.h"
#include "descry.h"
#include "distance.h"
#include "knn.h"

namespace suo15features {
    typedef pair<size_t, size_t> CP;

    struct Matcher_options{
        math_tools::Distance_options distance_options;
        float lowe_ratio_threshold;
        float distance_threshold;

        Matcher_options(){}
        Matcher_options(const math_tools::Distance_options& dist_opts,
                        float lowe_ratio, float dist_threshold):
                        distance_options(dist_opts),
                        lowe_ratio_threshold(lowe_ratio), distance_threshold(dist_threshold){}
    };

    struct Result{
        vector<int> matches_1_2;
        vector<int> matches_2_1;
    };

    class Matcher {
    public:
        Matcher(){}
        virtual vector<CP> GetMatchedKeypoints(
                const vector<cv::KeyPoint>& keypoints_1,
                cv::Mat& descriptors_1,
                const vector<cv::KeyPoint>& keypoints_2,
                cv::Mat& descriptors_2);
    };
}


#endif //LUT15VO_MATCHER_H
