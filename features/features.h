//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_FEATURES_H
#define LUT15VO_FEATURES_H

#include "common_include.h"

namespace suo15features {
    class Features {
        //keypoints, descriptor
    protected:
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;//一般对应的keypoints的向量
        //是哪一种的features,要标定一下
    public:
        Features();
        Features(const vector<cv::KeyPoint>& kps, const cv::Mat& descs);
    };
}


#endif //LUT15VO_FEATURES_H
