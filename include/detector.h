//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_EXTRACTOR_H
#define LUT15VO_EXTRACTOR_H

#include "common_include.h"

namespace suo15features {


    class ExtractorNode{
    public:
        vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;

        ExtractorNode():bNoMore(false){}
        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
    };

    template <typename T>
    class Detector {//表示要提取什么样的特征点，自定义的还是官方的cv::KeyPoints
    public:
        Detector<T>(){}
        virtual vector<cv::Mat> GetImagePyramid(){
            return vector<cv::Mat>(0);
        }
        virtual vector<int> GetKeypointsLevels(){
            return vector<int>(0);
        }
        virtual vector<T> ExtractorKeyPoints(const cv::Mat& ori_img)
        {
            cout<<"Father Detector func!"<<endl;
            return vector<T>();
        }
    };
}

#endif //LUT15VO_EXTRACTOR_H
