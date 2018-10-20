//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESC_S128_H
#define LUT15VO_DESC_S128_H

#include "descriptor.h"

namespace suo15features {
    class Desc_s128 : public Descriptor{
    public:
        struct Octave{
            typedef vector<cv::Mat> ImageVector;
            ImageVector img;
            ImageVector dog;
            ImageVector grad;
            ImageVector ori;
        };
        typedef vector<Octave> Octaves;

    private:
        Octaves octaves;
    public:
        //这里可以是一层的image也可以是多层的octaves
        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, const vector<cv::KeyPoint>& keypoints);
        //cost same time to create the img pyramid
        void descriptor_generation(void);
        void generate_grad_ori_images(Octave* octave);
        void orientation_assignment(cv::KeyPoint& kp, const Octave* ocatve, vector<float>& orientations);
        bool descriptor_assignment(cv::KeyPoint& kp, cv::Mat& desc, const Ocatve* octave);

        float keypoint_relative_scale(const cv::KeyPoint& kp);
        float keypoint_absolute_scale(const cv::KeyPoint& kp);
    };
}

#endif //LUT15VO_DESC_S128_H
