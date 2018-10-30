//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESC_S128_H
#define LUT15VO_DESC_S128_H

#include "descry.h"
#include "sift_keypoint.h"
#define MATH_PI         3.14159265358979323846264338327950288   // pi
#define MATH_SQRT2      1.41421356237309504880168872420969808   // sqrt(2)
namespace suo15features {
    struct S128_options{
        int num_samples_per_octave;
        int min_octave;
        int max_octave;
        float contrast_threshold;
        float base_blur_sigma;
        float inherent_blur_sigma;
        S128_options():num_samples_per_octave(3),
                       min_octave(0),
                       max_octave(4),
                       contrast_threshold(-1.0f),
                       base_blur_sigma(1.6f),
                       inherent_blur_sigma(0.5f){}
    };

    struct Simple_Octave{
        typedef vector<cv::Mat> ImageVector;
        ImageVector img;
        ImageVector grad;
        ImageVector ori;
    };
    typedef vector<Simple_Octave> Simple_Octaves;

    class Desc_s128 : public Descry<Sift_KeyPoint>{
    private:
        S128_options desc_s128_options;
        cv::Mat orig;
        cv::Mat descriptors;
        Descriptors define_descriptors;
        Simple_Octaves octaves;
    public:
        Desc_s128(S128_options options);

        void set_image(const cv::Mat& img);
        //这里可以是一层的image也可以是多层的octaves, keypoints 需要修改值的，因为需要修改keypoints的方向
        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, vector<Sift_KeyPoint>& keypoints);
        Descriptors process(const cv::Mat& image, vector<Sift_KeyPoint>& keypoints);

        //cost same time to create the img pyramid
        Simple_Octaves create_octaves(const cv::Mat& image);
        void add_octave(cv::Mat &image, float has_sigma, float target_sigma);

        void descriptor_generation(vector<Sift_KeyPoint>& keypoints);
        void generate_grad_ori_images(Simple_Octave* octave);
        void orientation_assignment(const Sift_KeyPoint& kp, const Simple_Octave* octave, vector<float>& orientations);
        bool descriptor_assignment(const Sift_KeyPoint& kp, Descriptor& desc, const Simple_Octave* octave);

        float keypoint_relative_scale(const Sift_KeyPoint& kp);
        float keypoint_absolute_scale(const Sift_KeyPoint& kp);
    };
}

#endif //LUT15VO_DESC_S128_H
