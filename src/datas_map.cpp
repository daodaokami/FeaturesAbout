//
// Created by lut on 18-12-25.

#include <iostream>
#include "datas_map.h"

KeyPoints_Map::KeyPoints_Map(int preSize){
    this->pre_size = pre_size;
    this->unm_v_ikps.reserve(this->pre_size);
}

void KeyPoints_Map::CreateUnorder_Map_KpIndex_vecIndex_IndexKP(vector<cv::KeyPoint> &kps) {
    int count = 0;
    for(int i=0; i<kps.size(); i++) {
        kpIndex index = compress_func_1(kps[i].pt.x, kps[i].pt.y);
        //向unordermap中插入数据
        if(this->unm_v_ikps.find(index) == this->unm_v_ikps.end()){
            //是空的
            vector<Index_Kp>v_ikps;
            v_ikps.reserve(3);//只是一个假设的空间大小
            v_ikps.push_back(Index_Kp(i, &kps[i]));
            this->unm_v_ikps.insert(make_pair(index, v_ikps));
        }
        else{
            //因为可能有多层可能是不一样的点,所以存在一样的index的点
            count ++;
            //std::cout<<this->unm_v_ikps[index].size()<<std::endl;
            //cout<<"cur_index "<< index <<endl;
            //到重复的点了就不可能是空
            bool flag_insert = true;
            for(int j=0; j<this->unm_v_ikps[index].size(); j++){
                //查看索引的大小,一般都是1
                if(unm_v_ikps[index][j].ptr_kp->pt.x == kps[i].pt.x &&
                        unm_v_ikps[index][j].ptr_kp->pt.y == kps[i].pt.y &&
                        unm_v_ikps[index][j].ptr_kp->octave == kps[i].octave)
                {
                    flag_insert = false;
                    cout<<unm_v_ikps[index][j].ptr_kp->pt.x<<"-"<<kps[i].pt.x<<endl
                        <<unm_v_ikps[index][j].ptr_kp->pt.y<<"-"<<kps[i].pt.y<<endl
                        <<unm_v_ikps[index][j].ptr_kp->octave<<"-"<<kps[i].octave<<endl
                        <<unm_v_ikps[index][j].ptr_kp->response<<"-"<<kps[i].response<<endl
                        <<unm_v_ikps[index][j].ptr_kp->class_id<<"-"<<kps[i].class_id<<endl;
                    cerr<<"the same !"<<endl;
                }
            }
            if(flag_insert)
                unm_v_ikps[index].push_back(Index_Kp(i, &kps[i]));
            //一些点的位置是一样的,但是因为octave不同,所以表示不同层之间的数据
        }
    }
    cout<<"count size is "<<count<<endl;
    //48 次重复
}

//返回的是在原始的kps_2中的序列号
KeyPoints_Map::vecIndex KeyPoints_Map::GetKeyPointsIndex(cv::KeyPoint& kp){
    kpIndex kp_index = compress_func_1(kp.pt.x, kp.pt.y);
    if(this->unm_v_ikps.find(kp_index) != this->unm_v_ikps.end()) {
        for (int j = 0; j < this->unm_v_ikps[kp_index].size(); j++){
            //查看索引的大小,一般都是1
            if(unm_v_ikps[kp_index][j].ptr_kp->pt.x == kp.pt.x &&
                unm_v_ikps[kp_index][j].ptr_kp->pt.y == kp.pt.y &&
                unm_v_ikps[kp_index][j].ptr_kp->octave == kp.octave){
                return unm_v_ikps[kp_index][j].vec_index;
            }
        }
    }
    return -1;
}

void KeyPoints_Map::CreateUnorder_Map_OuterIndex_InnerIndex(vector<cv::KeyPoint>& keyps){
    for(int outerIndex=0; outerIndex<keyps.size(); outerIndex++){
        kpIndex kp_index = compress_func_1(keyps[outerIndex].pt.x, keyps[outerIndex].pt.y);
        if(this->unm_v_ikps.find(kp_index) != this->unm_v_ikps.end()){
            for(int vindex =0; vindex<this->unm_v_ikps[kp_index].size(); vindex++){
                if(unm_v_ikps[kp_index][vindex].ptr_kp->pt.x == keyps[outerIndex].pt.x &&
                   unm_v_ikps[kp_index][vindex].ptr_kp->pt.y == keyps[outerIndex].pt.y &&
                   unm_v_ikps[kp_index][vindex].ptr_kp->octave == keyps[outerIndex].octave){
                    //这里存储的点与当前的一样
                    if(this->unm_v_outerIndex2innerIndex.find(outerIndex) == this->unm_v_outerIndex2innerIndex.end()) {//这里要是出现了重复的怎么办?不好,不如直接用map存储
                        this->unm_v_outerIndex2innerIndex.insert(
                        //通过map,映射这两个空间
                        make_pair(outerIndex, unm_v_ikps[kp_index][vindex].vec_index));//表示的当前的pt 在kps中的索引序号
                    }
                    else{
                        //是不可能出现的,outerIndex是不会重复的!
                    }
                }
            }
            //判断v_ikps中是否存在一样的点,理论上都是存在的,要不然,就是keyps中的数据有误
        }
        else{
            cerr<<"can not find this kp index !!!"<<endl;
        }
    }
}
/*
 *
 * 当前的数据结构:    unordered_map<KpIndex, vector<Index_Kp>> unm_v_ikps;
 *                               通过点的位置信息得到索引
 *                                        存储一个向量,1.vec_index 2.指向的特征点的值
 *
 * */