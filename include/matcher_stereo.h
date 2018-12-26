//
// Created by lut on 18-11-7.
//

#ifndef LUT15VO_MATCHER_STEREO_H
#define LUT15VO_MATCHER_STEREO_H

/*
 * 用途是用来计算点的深度信息,能够进行平面的计算与恢复
 * 是接下来实现的中要功能点之一
 * the most import is get the points at supported area;
 * 1.对图像的特征点开始就会有划分!!要实现一个函数能够快速得到摸一个区域内的特征点集
 * */

#include "opencv2/core/core.hpp"
#include <vector>
namespace suo15features {
    struct Stereo_options{
        int TH_LOW;
        int TH_HIGH;
        int HISTO_LENGTH;
        Stereo_options(int low = 50, int high = 100, int histo = 30):TH_LOW(low), TH_HIGH(high),
                HISTO_LENGTH(histo){}
    };

    class Matcher_stereo {
    public:
        Matcher_stereo():img_sz(cv::Size()){}
        Matcher_stereo(const Stereo_options& opts, const cv::Size sz, const std::vector<cv::KeyPoint>& lkps,
                       const std::vector<cv::KeyPoint>& rkps, const cv::Mat& ldescs, const cv::Mat& rdescs,
                       const float baseline, const float fx, const std::vector<float>& mvscaleFactors
        ):options(opts), img_sz(sz), left_keypoints(lkps), right_keypoints(rkps),
          left_descriptors(ldescs), right_descriptors(rdescs),
          mb(baseline), mvScaleFactors(mvscaleFactors)
        { mbf = mb*fx; }

        void setImagePyramid(const std::vector<cv::Mat>& limgpyr, const std::vector<cv::Mat>& rimgpyr){
            this->left_imagePyramid = limgpyr;
            this->right_imagePyramid = rimgpyr;
        }
        void createRowIndexes();
        void ComputeStereoMatches();

        std::vector<int> GetMatches();
        std::vector<int> GetDistance();
    protected:
        cv::Size img_sz;//主要就是样行数
        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> right_keypoints;
        cv::Mat left_descriptors;
        cv::Mat right_descriptors;

        std::vector<float> mvScaleFactors;
        std::vector<std::vector<size_t>> vRowIndexes;

        std::vector<int> vMatches;
        std::vector<int> vDistance;//binary

        std::vector<float> mvuRight;
        std::vector<float> mvDepth;
        float mb, mbf;//基线的长度
        Stereo_options options;

        std::vector<cv::Mat> left_imagePyramid;
        std::vector<cv::Mat> right_imagePyramid;
    };

    inline std::vector<int> Matcher_stereo::GetMatches() { return vMatches; }
    inline std::vector<int> Matcher_stereo::GetDistance() { return vDistance; }
}

#endif //LUT15VO_MATCHER_STEREO_H
