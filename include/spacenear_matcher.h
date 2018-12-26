//
// Created by lut on 18-12-22.
//

#ifndef SPATIALSUBDIVISION_SPACENEAR_MATCHER_H
#define SPATIALSUBDIVISION_SPACENEAR_MATCHER_H

#include "spatical_subdivision.h"

class SpaceNear_Matcher{
private:
    //输入是图像大小,特征点,对应的描述子,返回的是匹配的DMatches
    //首先要得到特征点之间的网格关系
    GridNet* gridNet_1, * gridNet_2;

public:
    SpaceNear_Matcher():gridNet_1(NULL), gridNet_2(NULL){}
    SpaceNear_Matcher(GridNet* gridNet_1, GridNet* gridNet_2);
    void SetGridNet(GridNet* gridNet_1, GridNet* gridNet_2){
        this->gridNet_1 = gridNet_1;
        this->gridNet_2 = gridNet_2;
    }

    void Set_KeyPoints_1(int index, std::vector<cv::KeyPoint>& kps_1);
    void Set_KeyPoints_2(int index, std::vector<cv::KeyPoint>& kps_2);

    void merge_kps_descs(std::vector<std::vector<cv::KeyPoint*>*> *ptr_v_ptr_vkps, std::vector<cv::Mat>& descs,
                         std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs);
    void merge_kps_descs(std::vector<cv::KeyPoint*> *ptr_vkps, cv::Mat& descs, std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs);

    void SpaceNearMatcher(int index, std::vector<cv::DMatch>& sub_matches);
    void SpaceNearMatcher(std::vector<cv::KeyPoint>& keyps_1, std::vector<cv::KeyPoint>& keyps_2,std::vector<cv::DMatch>& all_matches);
    void drawMatches(cv::Mat& img1, std::vector<cv::KeyPoint>& kps1,
                     cv::Mat& img2, std::vector<cv::KeyPoint>& kps2,
                     std::vector<cv::DMatch>& matches);
};
#endif //SPATIALSUBDIVISION_SPACENEAR_MATCHER_H
