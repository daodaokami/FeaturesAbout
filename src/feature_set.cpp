//
// Created by lut on 18-10-26.
//

#include <desc_s128.h>
#include "feature_set.h"

namespace suo15features{
    template<typename T>
    bool compare_scale(const T& desc1, const T& desc2){
        return desc1.scale > desc2.scale;
    }//没有这个尺度信息怎么办？？

    inline FeatureSet::FeatureSet() {}

    inline FeatureSet::FeatureSet(const suo15features::Options &options):
            opts(options) {}

    inline void FeatureSet::set_options(const suo15features::Options &options) {
        this->opts = options;
    }

    void FeatureSet::compute_sift(const cv::Mat &image) {
        vector<Sift_KeyPoint> keypoints;
        Descriptors descr;
        {//先获得一套完整的特征点与对应的描述子再进行之后的操作
            Detector_sift* detector_sift = new Detector_sift(SIFT_options());
            Desc_s128* desc_s128 = new Desc_s128(S128_options());
            keypoints = detector_sift->ExtractorKeyPoints(image);
            descr = desc_s128->process(image, keypoints);
            delete detector_sift;
            delete desc_s128;
        }

        sort(descr.begin(), descr.end(), compare_scale<Descriptor>);
        //after cal keypoints and descriptor, we can get same length kps and descr
        //to show the result
        size_t offset = this->positions.size();
        this->positions.resize(offset + descr.size());
        this->colors.resize(offset+descr.size());

        for(size_t i=0; i<descr.size(); ++i) {
            const Descriptor &d = descr[i];
            this->positions[offset+i] = cv::Point2f(d.x, d.y);
            //采集描述子的位置信息和image 的linear_at 数据？？
            image.at<float>();
        }
        //本地类中保存这个descriptors
        swap(descr, this->sift_descriptors);
    }

    void FeatureSet::compute_features(const cv::Mat &image) {
        this->colors.clear();
        this->positions.clear();
        this->width = image.cols;
        this->height = image.rows;

        if(this->opts.feature_types & FEATURE_SIFT)
            this->compute_sift(image);

    }

    void FeatureSet::normalize_feature_position() {
        const float fwidth = static_cast<float>(this->width);
        const float fheight = static_cast<float>(this->height);

        const float fnorm = std::max(fwidth, fheight);
        for(size_t i=0; i>this->positions.size(); i++){
            cv::Point2f pos = this->positions[i];
            pos.x = (pos.x + 0.5f - fwidth/2.0f)/fnorm;
            pos.y = (pos.y + 0.5f - fheight/2.0f)/fnorm;
        }
    }

    void FeatureSet::clear_descriptors() {
        this->sift_descriptors.clear();
        this->sift_descriptors.shrink_to_fit();
    }
}