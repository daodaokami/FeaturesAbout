//
// Created by lut on 18-12-22.
//

#include <spacenear_matcher.h>
#include <opencv2/features2d.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <chrono>

SpaceNear_Matcher::SpaceNear_Matcher(GridNet *gridNet_1, GridNet *gridNet_2) {
    this->gridNet_1 = gridNet_1;
    this->gridNet_2 = gridNet_2;
}

void SpaceNear_Matcher::SpaceNearMatcher(std::vector<cv::KeyPoint>& keyps_1, std::vector<cv::KeyPoint>& keyps_2,
                                         std::vector<cv::DMatch> &all_matches) {
    // SpaceNear_Matcher
    int grid_size = this->gridNet_1->gridKeypoints.size();
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    //得到对点内的匹配关系
    //还有一个问题是,序列的关系!!!
    //首先最简单的是重新,重新计算keypoints 和 descriptors得到,都是新的数据,并绘制出来查看效果
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    cv::Mat descriptors_1, descriptors_2;
    for(int index=0; index<grid_size; index++){
        std::vector<cv::KeyPoint*> *ptr_vkps_1 = &this->gridNet_1->gridKeypoints[index];//获取了第index块的当前特征点
        cv::Mat ptr_descs_1 = this->gridNet_1->gridDescs[index];//本身就是浅拷贝

        //找img2 中的相邻特征点与相邻的描述子
        std::vector<std::vector<cv::KeyPoint*>*> v_ptr_vkps_2;
        int neighbor_size = this->gridNet_2->neighbors[index].size();
        std::vector<cv::Mat> v_descs_2;
        v_ptr_vkps_2.resize(neighbor_size);
        v_descs_2.resize(neighbor_size);
        for(int neighbor = 0; neighbor<neighbor_size; neighbor++){
            v_ptr_vkps_2[neighbor] = this->gridNet_2->neighbors[index][neighbor];
            v_descs_2[neighbor] = this->gridNet_2->descs_neighbors[index][neighbor];
        }
        //把这些分散的数据整合成一串的数据
        std::vector<cv::KeyPoint> kps_1, kps_2;
        cv::Mat desc_1, desc_2;

        merge_kps_descs(ptr_vkps_1, ptr_descs_1, kps_1, desc_1);
        merge_kps_descs(&v_ptr_vkps_2, v_descs_2, kps_2, desc_2);
        keyps_1.insert(keyps_1.end(), kps_1.begin(), kps_1.end());
        keyps_2.insert(keyps_2.end(), kps_2.begin(), kps_2.end());
        //一次插入一个块内的特征点
        descriptors_1.push_back(desc_1);
        descriptors_2.push_back(desc_2);
        //这种时候是存在误匹配的
        std::cout<<index<<" SpaceNearMatcher kps1 "<<kps_1.size()<<" desc_1 "<<desc_1.size<<std::endl;
        std::cout<<index<<" SpaceNearMatcher kps2 "<<kps_2.size()<<" desc_2 "<<desc_2.size<<std::endl;
        //这里没有出错,应该是set keypoints 出错了
        std::vector<cv::DMatch> sub_matches;
        sub_matches.clear();
        if(desc_1.empty() || desc_2.empty()){
            continue;
        }
        matcher.match(desc_1, desc_2, sub_matches);
        //在vector的matchespush中,需要注意的是,可能存在
        int start_posi_1 = keyps_1.size() - kps_1.size();
        int start_posi_2 = keyps_2.size() - kps_2.size();

        for(int i=0; i<sub_matches.size(); i++){
            std::cout<<"cur_sub_matches "<<i<<"  "<<sub_matches[i].queryIdx<<"   -   "<<sub_matches[i].trainIdx<<std::endl;
            sub_matches[i].queryIdx += start_posi_1;
            sub_matches[i].trainIdx += start_posi_2;
            //这里肯定是错的,要完成的任务如下
            /**
             * sub_matches[] 是当前训练的A-的索引和对应B-的索引的块内的索引号
             *
             *
             * */

        }
        all_matches.insert(all_matches.end(), sub_matches.begin(), sub_matches.end());
    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_submatches = end - start;
    std::cout<<"all matches cost "<<elapsed_seconds_submatches.count()<<std::endl;
    std::cout<<"all matches size "<<all_matches.size()<<std::endl;
}

void SpaceNear_Matcher::SpaceNearMatcher(int index, std::vector<cv::DMatch> &sub_matches) {
    std::vector<cv::KeyPoint*> *ptr_vkps_1 = &this->gridNet_1->gridKeypoints[index];//获取了第index块的当前特征点
    cv::Mat ptr_descs_1 = this->gridNet_1->gridDescs[index];//本身就是浅拷贝

    //找img2 中的相邻特征点与相邻的描述子
    std::vector<std::vector<cv::KeyPoint*>*> v_ptr_vkps_2;
    int neighbor_size = this->gridNet_2->neighbors[index].size();
    std::vector<cv::Mat> v_descs_2;
    v_ptr_vkps_2.resize(neighbor_size);
    v_descs_2.resize(neighbor_size);
    for(int neighbor = 0; neighbor<neighbor_size; neighbor++){
        v_ptr_vkps_2[neighbor] = this->gridNet_2->neighbors[index][neighbor];
        v_descs_2[neighbor] = this->gridNet_2->descs_neighbors[index][neighbor];
    }
    //把这些分散的数据整合成一串的数据
    std::vector<cv::KeyPoint> kps_1, kps_2;
    cv::Mat desc_1, desc_2;

    merge_kps_descs(ptr_vkps_1, ptr_descs_1, kps_1, desc_1);
    merge_kps_descs(&v_ptr_vkps_2, v_descs_2, kps_2, desc_2);
    std::cout<<"SpaceNearMatcher kps1 "<<kps_1.size()<<" desc_1 "<<desc_1.size<<std::endl;
    std::cout<<"SpaceNearMatcher kps2 "<<kps_2.size()<<" desc_2 "<<desc_2.size<<std::endl;
    //这里没有出错,应该是set keypoints 出错了
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    matcher.match(desc_1, desc_2, sub_matches);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_submatches = end - start;
    std::cout<<"sub matches cost "<<elapsed_seconds_submatches.count()<<std::endl;
}

void SpaceNear_Matcher::Set_KeyPoints_1(int index, std::vector<cv::KeyPoint> &kps_1) {
    std::vector<cv::KeyPoint*> *ptr_vkps_1 = &this->gridNet_1->gridKeypoints[index];//获取了第index块的当前特征点
    int kps_size = ptr_vkps_1->size();
    kps_1.resize(kps_size);
    for(int i=0; i<kps_size; i++) {
        cv::KeyPoint kp;
        kp.pt = (*ptr_vkps_1)[i]->pt;
        kp.octave = (*ptr_vkps_1)[i]->octave;
        kp.angle = (*ptr_vkps_1)[i]->angle;
        kp.response = (*ptr_vkps_1)[i]->response;
        kps_1[i] = kp;
    }
    std::cout<<"Set_Kps1 "<<kps_1.size()<<std::endl;
}

void SpaceNear_Matcher::Set_KeyPoints_2(int index, std::vector<cv::KeyPoint> &kps_2) {
    std::vector<std::vector<cv::KeyPoint*>*> v_ptr_vkps_2;
    int neighbor_size = this->gridNet_2->neighbors[index].size();

    v_ptr_vkps_2.resize(neighbor_size);
    for(int neighbor = 0; neighbor<neighbor_size; neighbor++){
        v_ptr_vkps_2[neighbor] = this->gridNet_2->neighbors[index][neighbor];
    }

    int sum_kps = 0;
    for(int i=0; i<v_ptr_vkps_2.size(); i++){
        sum_kps += v_ptr_vkps_2[i]->size();
    }
    kps_2.resize(sum_kps);
    int count = 0;
    for(int i=0; i<v_ptr_vkps_2.size(); i++) {
        //即存在多少的邻居!!!
        for (int j = 0; j < v_ptr_vkps_2[i]->size(); j++) {
            cv::KeyPoint kp;
            kp.pt = (*(v_ptr_vkps_2[i]))[j]->pt;
            kp.octave = (*(v_ptr_vkps_2[i]))[j]->octave;
            kp.angle = (*(v_ptr_vkps_2[i]))[j]->angle;
            kp.response = (*(v_ptr_vkps_2[i]))[j]->response;
            kps_2[count] = kp;
            count++;
        }
    }
    std::cout<<"Set Kps2 "<<kps_2.size()<<std::endl;
}
void SpaceNear_Matcher::drawMatches(cv::Mat &img1, std::vector<cv::KeyPoint> &kps1, cv::Mat &img2,
                                    std::vector<cv::KeyPoint> &kps2, std::vector<cv::DMatch> &matches) {

    this->Set_KeyPoints_1(17, kps1);
    this->Set_KeyPoints_2(17, kps2);
    //采用多线程,四路同时计算
    cv::Mat img_kps1;
    cv::drawKeypoints(img1, kps1, img_kps1);
    cv::Mat img_kps2;
    cv::drawKeypoints(img2, kps2, img_kps2);
    //从 第 85 号元素开始,就没有被赋予值了!!
    cv::imshow("drawmatches_img_kps1", img_kps1);
    cv::imshow("drawmatches_img_kps2", img_kps2);
    cv::Mat img_matches;
    cv::drawMatches(img1, kps1, img2, kps2, matches, img_matches);
    cv::imshow("draw_matches_img_matches", img_matches);
    cv::waitKey(0);
}

void SpaceNear_Matcher::merge_kps_descs(std::vector<cv::KeyPoint*> *ptr_vkps, cv::Mat& descs, std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs){
    cdescs = descs;
    kps.resize(ptr_vkps->size());
    for(int i=0; i<ptr_vkps->size(); i++){
        cv::KeyPoint kp;
        kp.pt = (*ptr_vkps)[i]->pt;
        kp.octave = (*ptr_vkps)[i]->octave;
        kp.angle = (*ptr_vkps)[i]->angle;
        kp.response = (*ptr_vkps)[i]->response;
        kps[i] = kp;
    }
    //完成融合与构造了

}

void SpaceNear_Matcher::merge_kps_descs(std::vector<std::vector<cv::KeyPoint*>*> *ptr_v_ptr_vkps, std::vector<cv::Mat>& descs,
                        std::vector<cv::KeyPoint>& kps, cv::Mat& cdescs){
    int sum_kps = 0;
    for(int i=0; i<ptr_v_ptr_vkps->size(); i++){
        sum_kps += (*ptr_v_ptr_vkps)[i]->size();
    }
    kps.resize(sum_kps);
    int count = 0;
    for(int i=0; i<ptr_v_ptr_vkps->size(); i++){
        //即存在多少的邻居!!!
        for(int j=0; j<(*ptr_v_ptr_vkps)[i]->size(); j++){
            cv::KeyPoint kp;
            kp.pt = (*(*ptr_v_ptr_vkps)[i])[j]->pt;
            kp.octave = (*(*ptr_v_ptr_vkps)[i])[j]->octave;
            kp.angle = (*(*ptr_v_ptr_vkps)[i])[j]->angle;
            kp.response = (*(*ptr_v_ptr_vkps)[i])[j]->response;
            kps[count] = kp;
            count ++;
        }
        cdescs.push_back(descs[i]);
    }
}
