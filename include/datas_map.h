//
// Created by lut on 18-12-25.
//

#ifndef SPATIALSUBDIVISION_DATAS_MAP_H
#define SPATIALSUBDIVISION_DATAS_MAP_H

#include <unordered_map>
#include <map>
#include <vector>
#include <opencv2/core/types.hpp>
using namespace std;

struct Index_Kp{
    int vec_index;
    cv::KeyPoint* ptr_kp;
    Index_Kp(int i, cv::KeyPoint* ptrKp){
        vec_index = i;
        ptr_kp = ptrKp;
    }
};
class KeyPoints_Map{
private:
    typedef int kpIndex;
    typedef int vecIndex;
    int pre_size;
    unordered_map<kpIndex, vector<Index_Kp>> unm_v_ikps;  //有一定的可能会遇见重复的元素,所以在唯一位置的时候,用一个vector 存储这个index和对应的keypoint
                                                        //大多数的大小都是1,很难出现都一样的情况
                                                        //这个容器本身的大小,一开始就可以确定的

    //这个map用于存储,当在keyps_2(18000,有很多重复点的),其中的序号与kps_2(2000,无重复点),一一对应的映射关系
    map<vecIndex, vecIndex> unm_v_outerIndex2innerIndex;
public:
    KeyPoints_Map(){}
    KeyPoints_Map(int preSize);

    unordered_map<kpIndex, vector<Index_Kp>>* GetMap(){
        return &this->unm_v_ikps;
    };

    map<vecIndex, vecIndex>* GetIndexMap(){
        return &this->unm_v_outerIndex2innerIndex;
    };

    kpIndex compress_func_1(float x, float y){
        return static_cast<int>(round(x+y+x*y));
    }//重复的概率应该是比较小的

    //得到特征点在keyps_2中的位置
    vecIndex GetKeyPointsIndex(cv::KeyPoint& kp);
    void CreateUnorder_Map_KpIndex_vecIndex_IndexKP(vector<cv::KeyPoint>& kps);

    void CreateUnorder_Map_OuterIndex_InnerIndex(vector<cv::KeyPoint>& keyps);
};

/*
 * 要快速的从DMatches的向量结构转换成map结构
 * 解决的问题是,DMatches中train的序号中,对应
 * 的keyps_2中的元素应该对应原版的kps_2中的序
 * 号
 * */
class DMatches_Map{
private:

public:

};
#endif //SPATIALSUBDIVISION_DATAS_MAP_H
