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

    class Detector {
    public:
        Detector();
        virtual vector<cv::KeyPoint> ExtractorKeyPoints(const cv::Mat& ori_img);
    };
}

#endif //LUT15VO_EXTRACTOR_H
