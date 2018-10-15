//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESC_S256_H
#define LUT15VO_DESC_S256_H

#include "detector_orb.h"
#include "descriptor.h"
#include "desc_b256.h"

namespace suo15features {
    class Desc_b256 :public Descriptor{
    protected:
        std::vector<cv::Point> pattern;
        Detector* detector;
    public:
        Desc_b256(Detector* ptr);
        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, const vector<cv::KeyPoint>& keypoints);
    };
}

#endif //LUT15VO_DESC_S256_H
