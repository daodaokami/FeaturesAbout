//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_MATCHER_KNN_H
#define LUT15VO_MATCHER_KNN_H

#include "matcher.h"
#include "knn.h"

namespace suo15features {
    class Matcher_knn:public Matcher{
    private:
        Matcher_options matcher_options;
    public:
        Matcher_knn(){}
        Matcher_knn(const Matcher_options& matcherOptions);

        void twoway_match(cv::Mat& set_1, size_t set_1_size,
                          cv::Mat& set_2, size_t set_2_size,
                          Result& matches);

        void oneway_match(cv::Mat& set_1, int set_1_size,
                          cv::Mat& set_2, int set_2_size,
                          vector<int>& result);

        virtual vector<CP> GetMatchedKeypoints(
                const vector<cv::KeyPoint>& keypoints_1,
                cv::Mat& descriptors_1,
                const vector<cv::KeyPoint>& keypoints_2,
                cv::Mat& descriptors_2);
    };
}

#endif //LUT15VO_KNN_MATCHER_H
