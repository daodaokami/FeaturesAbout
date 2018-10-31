//
// Created by lut on 18-10-26.
//

#include "feature_set.h"

namespace suo15features{
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

    void FeatureSet::compute_orb(const cv::Mat &image) {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        {
            Detector_orb* detector_orb = new Detector_orb(this->feature_options.orb_opts);
            Desc_b256* desc_b256 = new Desc_b256(detector_orb);
            vector<cv::KeyPoint> keypoints = detector_orb->ExtractorKeyPoints(image);
            this->descriptors = desc_b256->ComputeDescriptor(image, keypoints);
            this->positions.resize(keypoints.size());
            for(size_t i=0; i<keypoints.size();i++){
                this->positions[i] = keypoints[i].pt;
            }
            delete detector_orb;
            delete desc_b256;
        }
    }

    void FeatureSet::compute_features(const cv::Mat &image) {
        this->colors.clear();
        this->positions.clear();
        this->width = image.cols;
        this->height = image.rows;

        switch(this->feature_options.feature_types){
            case FEATURE_SIFT:
                this->compute_sift(image);
                break;
            case FEATURE_ORB:
                this->compute_orb(image);
                break;
            default:
                return ;
        }
        //计算出了descriptors
    }

    /* 不一定用的上这个函数 */
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