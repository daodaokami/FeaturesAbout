//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DESCRIPTOR_H
#define LUT15VO_DESCRIPTOR_H

#include "common_include.h"
#include "sift_keypoint.h"

namespace suo15features {
    struct Descriptor{
        float x;
        float y;
        float scale;
        float orientation;
        //这是描述子向量
        cv::Mat data;
        Descriptor(){}
        Descriptor(const cv::Mat& desc):data(desc){}
        Descriptor(float x, float y, float scale, float orientation, const cv::Mat& desc):
            x(x), y(y), orientation(orientation), data(desc){}
    };

    typedef vector<Descriptor> Descriptors;

    template <typename T>
    class Descry {//描述子中计算的Keypoints的属性类型
        //因为能够派生出各种类型的描述子
                //orb256
                //sift128
                //surf64
        //那么descriptor主要提供的是一个通用的接口，能够提供方便的子类的描述子提取
    public:
        Descry<T>(){}
        //这里注意的是需要 keypoints 是需要进行修改的，对其的方向等描述方向
        virtual cv::Mat ComputeDescriptor(const cv::Mat& image, vector<T>& keypoints);
        //virtual cv::Mat ComputeDescriptor(const cv::Mat& image, vector<Sift_KeyPoint>& keypoints);
    };

    template <typename T>
    cv::Mat Descry<T>::ComputeDescriptor(const cv::Mat& image, vector<T>& keypoints){
        cout<<"Descriptor father!"<<endl;
        return Mat();
    }

}

#endif //LUT15VO_DESCRIPTOR_H
