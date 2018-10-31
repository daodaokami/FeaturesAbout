//
// Created by lut on 18-10-26.
//

#include <iostream>
#include <detector_sift.h>
#include <desc_s128.h>
#include <sift_keypoint.h>
#include <matcher.h>
#include <feature_set.h>
#include <opencv2/features2d.hpp>
#include <matcher_knn.h>
#include "common_include.h"

using namespace std;

void ShowImgWithKeypoints(const string winname, const cv::Mat& img, const vector<cv::Point2f>& positions);
int main(int argc, char* argv[]){
    if(argc < 3){
        cerr<<"syntax: "<<argv[0] << " image1 image2 "<<endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    cv::imshow("img_1", img_1);
    cv::imshow("img_2", img_2);
    cv::waitKey(0);
    suo15features::Feature_options feature_options(suo15features::FeatureTypes::FEATURE_ORB,
        suo15features::SIFT_options(),
        suo15features::ORB_options(31, 15, 19, 1000, 1.2, 4, 20, 7),
        suo15features::Fast_options());

    //suo15features::A a;  //文件连接上了，但是featureSet为什么会出错？？

    suo15features::FeatureSet featureSet(feature_options);
    featureSet.compute_features(img_1);
    vector<cv::Point2f> position = featureSet.positions;
    cv::Mat descriptor = featureSet.descriptors;
    ShowImgWithKeypoints("out1", img_1, position);

    featureSet.compute_features(img_2);
    vector<cv::Point2f> position_2 = featureSet.positions;
    cv::Mat descriptor_2 = featureSet.descriptors;
    ShowImgWithKeypoints("out2", img_2, position_2);
    cv::waitKey(0);

    math_tools::Distance_options distance_options(256, math_tools::DistanceType::HAMMING_DISTANCE);
    suo15features::Matcher_options matcher_options(distance_options, 30, 0.8);
    suo15features::Matcher_knn matcher_knn(matcher_options);
    vector<int> result;
    matcher_knn.oneway_match(descriptor, descriptor.rows, descriptor_2, descriptor_2.rows, result);

    //或者直接调用opencv 提供的knn匹配算法
    
    return 0;
}

void ShowImgWithKeypoints(const string winname, const cv::Mat& img, const vector<cv::Point2f>& positions){
    vector<cv::KeyPoint> keypoints;
    keypoints.resize(positions.size());
    for(size_t i=0; i<positions.size(); i++){
        keypoints[i].pt = positions[i];
    }
    cv::Mat out;
    cv::drawKeypoints(img, keypoints, out);
    cv::imshow(winname, out);
}