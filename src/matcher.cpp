//
// Created by lut on 18-10-12.
//

#include "../include/matcher.h"

namespace suo15features{
    vector<CP> Matcher::GetMatchedKeypoints(const vector<cv::KeyPoint> &keypoints_1,
                                                              const cv::Mat &descriptors_1,
                                                              const vector<cv::KeyPoint> &keypoints_2,
                                                              const cv::Mat &descriptors_2) {
        cout<<"father matcher!"<<endl;
        return vector<pair<size_t, size_t>>();
    }
};