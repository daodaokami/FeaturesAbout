//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_H
#define LUT15VO_MATCHER_H

#include "common_include.h"
#include "descry.h"

namespace suo15features {
    struct Matcher_options{
        int descriptor_length;
        float lowe_ratio_threshold;
        float distance_threshold;
    };

    struct Result{
        vector<int> matches_1_2;
        vector<int> matches_2_1;
    };

    //这是可以相互搜索的匹配对
    typedef pair<size_t, size_t> CP;
    //CP是两个输入的keypoints的序号的链接
    class Matcher {
    public:
        void twoway_match(const Matcher_options& options,
                const Descriptors& set_1, size_t set_1_size,
                const Descriptors& set_2, size_t set_2_size,
                Result& matches);

        void oneway_match(const Matcher_options& options,
                const Descriptors& set_1, int set_1_size,
                const Descriptors& set_2, int set_2_size,
                vector<int>& result);

        virtual vector<CP> GetMatchedKeypoints(
                const vector<cv::KeyPoint>& keypoints_1,
                const cv::Mat& descriptors_1,
                const vector<cv::KeyPoint>& keypoints_2,
                const cv::Mat& descriptors_2);
    };
}


#endif //LUT15VO_MATCHER_H
