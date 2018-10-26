//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DETECTOR_SIFT_H
#define LUT15VO_DETECTOR_SIFT_H

#include "detector.h"
#include "sift_keypoint.h"
#include "common_include.h"

namespace suo15features {
    class Detector_sift : public Detector<Sift_KeyPoint>{
    public:
        struct SIFT_Options{
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
            inline SIFT_Options()
                    :num_samples_per_octave(3),
                     min_octave(0),
                     max_octave(4),
                     contrast_threshold(-1.0f),
                     edge_ratio_threshold(10.0f),
                     base_blur_sigma(1.6f),
                     inherent_blur_sigma(0.5f){
            }
        };

        typedef vector<Sift_KeyPoint> KeyPoints;
        struct Octave{
            typedef vector<cv::Mat> ImageVector;
            ImageVector img;
            ImageVector dog;
        };
        typedef vector<Octave> Octaves;
    protected:
        //一个Octave是存储的金字塔的一层！（S+3）
        void create_octaves(void);
        void add_octave(cv::Mat& image, float has_sigma, float target_sigma);
        void extrema_detection(void);
        size_t extrema_detection(cv::Mat s[3], int oi, int si);

        void keypoint_localization(void);

    public:
        explicit Detector_sift(SIFT_Options const& options);
        //set什么类型的Image都是一样的
        void set_image(const cv::Mat& img);
        void process(void);//process就是父类中virtual func
        virtual vector<Sift_KeyPoint> ExtractorKeyPoints(const cv::Mat& ori_img);

    private:
        cv::Mat orig;
        SIFT_Options sift_options;
        Octaves octaves;

        KeyPoints keypoints;
    public:
        ~Detector_sift();
    };
}

#endif //LUT15VO_SIFT_H
