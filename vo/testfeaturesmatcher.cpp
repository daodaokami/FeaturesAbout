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
#include "visualize.h"
#include "common_include.h"

using namespace std;

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
    suo15features::ShowImgWithKeypoints("out1", img_1, position);

    featureSet.compute_features(img_2);
    vector<cv::Point2f> position_2 = featureSet.positions;
    cv::Mat descriptor_2 = featureSet.descriptors;
    suo15features::ShowImgWithKeypoints("out2", img_2, position_2);
    cv::waitKey(0);

    math_tools::Distance_options distance_options(256, math_tools::DistanceType::HAMMING_DISTANCE);
    suo15features::Matcher_options matcher_options(distance_options, 30, 0.8);
    suo15features::Matcher_knn matcher_knn(matcher_options);
    vector<int> result;
    matcher_knn.oneway_match(descriptor, descriptor.rows, descriptor_2, descriptor_2.rows, result);
    suo15features::Result nn_res;
    matcher_knn.twoway_match(descriptor, descriptor.rows, descriptor_2, descriptor_2.rows, nn_res);
    //result表示的是当前的描述子对应的匹配的点，将这样的描述子对进行匹配
    //输出匹配的结果
    cout<<descriptor.rows<<" "<<descriptor_2.rows<<endl;
    cout<<result.size()<<endl;
    cout<<nn_res.matches_1_2.size()<<" "<<nn_res.matches_2_1.size()<<endl;
    cv::destroyAllWindows();

    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor, descriptor_2, matches);

    vector<vector<cv::DMatch>> knn_matches;
    vector<cv::DMatch> knn_goodMatches;
    matcher.knnMatch(descriptor, descriptor_2, knn_matches, 2);
    const float minRatio = 1.f / 1.5f;
    for(size_t i=0; i<knn_matches.size(); ++i){
        const cv::DMatch& bestMatch = knn_matches[i][0];
        const cv::DMatch& betterMatch = knn_matches[i][1];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            knn_goodMatches.push_back(bestMatch);
    }

    double min_dist = 10000, max_dist = 0;
    for(int i=0; i<descriptor.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    vector<cv::KeyPoint> keypoints_1(position.size());
    vector<cv::KeyPoint> keypoints_2(position_2.size());

    for(int i=0; i<keypoints_1.size(); ++i){
        keypoints_1[i].pt = position[i];
    }

    for(int i=0; i<keypoints_2.size(); ++i){
        keypoints_2[i].pt = position_2[i];
    }

    vector<cv::DMatch> good_matches;
    for(int i=0; i<descriptor.rows; ++i){
        if(matches[i].distance <= max(2*min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }

    cv::Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("good_match", img_goodmatch);
    cv::waitKey();

    cv::Mat img_knngoodMatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, knn_goodMatches, img_knngoodMatch);
    cv::imshow("knn_goodMatch", img_knngoodMatch);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
