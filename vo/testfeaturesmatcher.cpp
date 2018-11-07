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
#include <chrono>
#include "visualize.h"
#include "common_include.h"

using namespace std;
void showWidthImgMatch(string winname , const cv::Mat& image_1, const vector<cv::KeyPoint>& keypoints_1,
                       const cv::Mat& image_2, const vector<cv::KeyPoint>& keypoints_2,
                       const vector<cv::DMatch>& matches);

int main(int argc, char* argv[]){
    if(argc < 3){
        cerr<<"syntax: "<<argv[0] << " image1 image2 "<<endl;
        return 1;
    }
/**
 *
 * the one result is that extract keypoints and compute the descriptors just cost 20ms
 *
 * now the problem is just the descriptors creator it cost same error in same position!!
 *
 * */
    cv::Mat img_1 = cv::imread(argv[1], 0);
    cv::Mat img_2 = cv::imread(argv[2], 0);
/*计时 1*/
    suo15features::ORB_options orb_options(31, 15, 19, 1000, 1.2, 4, 20, 7);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    suo15features::Detector<cv::KeyPoint>* detector_orb = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints = detector_orb->ExtractorKeyPoints(img_1);
    suo15features::Descry<cv::KeyPoint>* orb_desc = new suo15features::Desc_b256(detector_orb);
    cv::Mat desc = orb_desc->ComputeDescriptor(img_1, keypoints);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;
    cout<<"desc1 row \n"<<desc.size<<"\n"<<desc<<endl;
    cv::Mat km1;
    cv::drawKeypoints(img_1, keypoints, km1);
    cv::imshow("km1", km1);
    cv::waitKey(0);

    for(int i=0; i<keypoints.size(); i++){
        cout<<keypoints[i].pt<<" "<<keypoints[i].angle<<endl;
    }
    suo15features::Detector<cv::KeyPoint>* detector_orb2 = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints2 = detector_orb2->ExtractorKeyPoints(img_2);
    suo15features::Descry<cv::KeyPoint>* orb_desc2 = new suo15features::Desc_b256(detector_orb2);
    cv::Mat desc2 = orb_desc2->ComputeDescriptor(img_2, keypoints2);
    cout<<"desc2 row \n"<<desc2.size<<"\n"<<desc2<<endl;
    cv::Mat km2;
    cv::drawKeypoints(img_2, keypoints2, km2);
    cv::imshow("km2", km2);
    cv::waitKey(0);

    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(desc, desc2, matches);

    vector<vector<cv::DMatch>> knn_matches;
    vector<cv::DMatch> knn_goodMatches;
    matcher.knnMatch(desc, desc2, knn_matches, 2);
    const float minRatio = 1.f / 2.0f;
    for(size_t i=0; i<knn_matches.size(); ++i){
        const cv::DMatch& bestMatch = knn_matches[i][0];
        const cv::DMatch& betterMatch = knn_matches[i][1];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            knn_goodMatches.push_back(bestMatch);
    }

    cv::Mat img_knngoodMatch;
    cv::drawMatches(img_1, keypoints, img_2, keypoints2, knn_goodMatches, img_knngoodMatch);
    cv::imshow("knn_goodMatch", img_knngoodMatch);
    cv::waitKey(0);

    showWidthImgMatch("wknn_goodmatch", img_1, keypoints, img_2, keypoints2, knn_goodMatches);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

void showWidthImgMatch(string winname, const cv::Mat& image_1, const vector<cv::KeyPoint>& keypoints_1,
                       const cv::Mat& image_2, const vector<cv::KeyPoint>& keypoints_2,
                       const vector<cv::DMatch>& matches){
    cv::Mat img_1t = image_1.t();
    cv::Mat img_2t = image_2.t();
    vector<cv::KeyPoint> kps1, kps2;
    kps1.resize(keypoints_1.size());
    kps2.resize(keypoints_2.size());
    for(size_t i=0; i<keypoints_1.size(); i++){
        kps1[i].pt.x = keypoints_1[i].pt.y;
        kps1[i].pt.y = keypoints_1[i].pt.x;
    }
    for(size_t i=0; i<keypoints_2.size(); i++){
        kps2[i].pt.x = keypoints_2[i].pt.y;
        kps2[i].pt.y = keypoints_2[i].pt.x;
    }
    cv::Mat out1;
    cv::drawMatches(img_1t, kps1, img_2t, kps2, matches, out1);
    cv::imshow(winname, out1.t());
}
