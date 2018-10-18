//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DETECTOR_SIFT_H
#define LUT15VO_DETECTOR_SIFT_H

#include "detector.h"
#include "Sift_KeyPoint.h"
#include "common_include.h"

namespace suo15features {
    class Detector_sift : public Detector{
    public:
        struct Options{
            //每层有效的DOG个数S，S=3，每阶需要的DOG图像个数是S+2,需要高斯平滑的图像个数是N = S+3
            int num_samples_per_octave;
            /*default to 0, which uses the input image size as base size.
             * values > 0, causes the image to be down scaled by factors of two
             * this can be set to -1, which expend the ori to 2*size
             * */
            int min_octave;
            int max_octave;
            //亚像素精度定位时的阈值
            float contrast_threshold;
            //消除边界响应，边界处的特征点不具有不变性
            float edge_ratio_threshold;
            //default sigma is 1.6
            float base_blur_sigma;
            //default is 0.5, inherernt blur sigma in the input image
            float inherent_blur_sigma;
            inline Options()
                    :num_samples_per_octave(3),
                     min_octave(0),
                     max_octave(4),
                     contrast_threshold(-1.0f),
                     edge_ratio_threshold(10.0f),
                     base_blur_sigma(1.6f),
                     inherent_blur_sigma(0.5f){
            }
        };
    protected:
        typedef vector<Sift_KeyPoint> KeyPoints;
        typedef cv::Mat Descriptors;
        KeyPoints keypoints;
        Descriptors descriptors;

        //一个Octave是存储的金字塔的一层！（S+3）
        struct Octave{
            typedef vector<cv::Mat> ImageVector;
            ImageVector img;
            ImageVector dog;
            ImageVector grad;
            ImageVector ori;
        };
        typedef vector<Octave> Octaves;
        void create_octaves(void);
        void add_octave(cv::Mat& image, float has_sigma, float target_sigma);
        void extrema_detection(void);
        size_t extrema_detection(cv::Mat s[3], int oi, int si);

        void keypoint_localization(void);
        void descriptor_generation(void);
        void generate_grad_ori_images(Octave* octave);

        void orientation_assignment(Sift_KeyPoint const& kp,
            Octave const* octave, vector<float>& orientations);
        bool descriptor_assignment(Sift_KeyPoint const& kp, cv::Mat& desc,
            Octave const* octave);

        float keypoint_relative_scale(const Sift_KeyPoint& kp);
        float keypoint_absolute_scale(const Sift_KeyPoint& kp);

    public:
        explicit Detector_sift(Options const& options);
        //set什么类型的Image都是一样的
        void set_image(const cv::Mat& img);
        void process(void);//process就是父类中virtual func

        KeyPoints const& get_keypoints() const;

        Descriptors const& get_Descriptors() const;

        virtual vector<cv::KeyPoint> ExtractorKeyPoints(const cv::Mat& ori_img);
    private:
        cv::Mat orig;
        Options sift_options;
        Octaves octaves;
    };

    inline vector<Sift_KeyPoint> const&
    Detector_sift::get_keypoints() const {
        return this->keypoints;
    }

    inline cv::Mat const&
    Detector_sift::get_Descriptors() const {
        return this->descriptors;
    }


}

#endif //LUT15VO_SIFT_H
