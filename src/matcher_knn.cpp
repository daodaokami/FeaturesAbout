//
// Created by lut on 18-10-12.
//

#include "../include/matcher_knn.h"

namespace suo15features {
    Matcher_knn::Matcher_knn(const Matcher_options &matcherOptions):matcher_options(matcherOptions){}

    void Matcher_knn::twoway_match(cv::Mat &set_1, size_t set_1_size,
                      cv::Mat &set_2, size_t set_2_size,
                      Result &matches){
                Matcher_knn::oneway_match(set_1, set_1_size,
                            set_2, set_2_size, matches.matches_1_2);

                Matcher_knn::oneway_match(set_2, set_2_size,
                            set_1, set_1_size, matches.matches_2_1);
    }

    void Matcher_knn::oneway_match(cv::Mat &set_1, int set_1_size,
                      cv::Mat &set_2, int set_2_size,
                      vector<int> &result){
        result.clear();
        result.resize(set_1_size, -1);
        if(set_1_size == 0 || set_2_size ==0)
            return ;
        //float const square_dist_thres = MATH_POW2(matcher_options.distance_threshold);
        math_tools::Knn<int> knn(this->matcher_options.distance_options, set_2_size);
        math_tools::Result<int> nn_res;
        for(int i=0; i<set_1_size; ++i){
            int* queryIdx = set_1.ptr<int32_t >(i);
            int* trainIdx = set_2.ptr<int32_t >(0);//从头开始，要检索素有的特征点
            knn.search_knnsample(queryIdx, trainIdx, &nn_res);
            cout<<"cur points is "<<i<<"\nbest is "<<nn_res.index_1st_best <<" dis is "<<nn_res.dist_2nd_best
                                       <<" second is "<<nn_res.index_2nd_best<<" dis is "<<nn_res.dist_2nd_best<<endl;
            /*
             * 1. 使用两中信息进行粗匹配
             * 2. 使用上点的空间距离信息,即记录好描述子,要选择二维图像空间中描述子距离最近的点
             *
             * */

        }
    }


    /*
     * before using this function must to
     * format the data to mat//orb direct use
     *                       //sift should use de sift.data.
     * */
    vector<CP> Matcher_knn::GetMatchedKeypoints(
            const vector<cv::KeyPoint> &keypoints_1,
            cv::Mat &descriptors_1,
            const vector<cv::KeyPoint> &keypoints_2,
            cv::Mat &descriptors_2){

    }
}