//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESC_S256_H
#define LUT15VO_DESC_S256_H

#include "detector_orb.h"
#include "descry.h"
#include "desc_b256.h"


namespace suo15features {

    class Desc_b256 : public Descry<cv::KeyPoint>{
    protected:
        std::vector<cv::Point> pattern;
        Detector<cv::KeyPoint>* detector;
        /*
         * 在母方法中添加获取尺度因子的方法
         * */
    public:
        Desc_b256(Detector<cv::KeyPoint>* ptr);

        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, vector<cv::KeyPoint>& keypoints);
    };
}

#endif //LUT15VO_DESC_S256_H
