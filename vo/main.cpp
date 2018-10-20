//
// Created by lut on 18-10-12.
//

#include "common_include.h"
#include <vector>
#include <detector.h>
#include <detector_fast.h>
#include <detector_orb.h>
#include "sift_keypoint.h"
#include <detector_sift.h>
#include <opencv2/features2d.hpp>
#include <descriptor.h>
#include <desc_b256.h>
#include <chrono>
#include <matcher_bf.h>

int main(){
    cv::Mat img_example = cv::imread("./data/example.jpeg", cv::IMREAD_ANYCOLOR);
    suo15features::Detector_sift::Options sift_options;
    suo15features::Detector_sift* detector_sift = new suo15features::Detector_sift(sift_options);
    vector<cv::KeyPoint> SKP = detector_sift->ExtractorKeyPoints(img_example);
    cv::Mat out;
    cv::drawKeypoints(img_example, SKP, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    /*
    cv::Mat img;
    string filepath = "./data/000001.png";

    img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    cout<<"here"<<endl;
    cout<<"here"<<endl;

    suo15features::Detector *detector;

    detector = new suo15features::Detector();
    cv::Mat m;
    detector->ExtractorKeyPoints(m);
    delete detector;
    detector = new suo15features::Detector_fast();
    vector<cv::KeyPoint> keypoints_fast = detector->ExtractorKeyPoints(img);
    delete detector;


    suo15features::ORB_config default_config(31, 15, 19, 1000, 1.2, 4, 20, 7);
    detector = new suo15features::Detector_orb(default_config);
    vector<int> mnkeypointsLevels = detector->GetKeypointsLevels();

    suo15features::Descriptor* descriptor;
    suo15features::Detector_orb detector_orb;
    //descriptor = new suo15features::Desc_b256();
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    vector<cv::KeyPoint> keypoints_orb = detector->ExtractorKeyPoints(img);

    descriptor = new suo15features::Desc_b256(detector);
    cv::Mat descs = descriptor->ComputeDescriptor(img, keypoints_orb);

    mnkeypointsLevels.clear();
    mnkeypointsLevels = detector->GetKeypointsLevels();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;
    delete detector;*/

    //将点展开，得到，分别得到不同层次上的特征点
/*    vector<vector<cv::KeyPoint>> allKeypoints;
    allKeypoints.resize(default_config._nlevels);
    for(size_t i=0; i<allKeypoints.size();i++){
        allKeypoints[i].reserve(default_config._nfeatures);
    }
    for(auto kp : keypoints){
        allKeypoints[kp.octave].push_back(kp);
    }
    for(size_t i=0; i<allKeypoints.size(); i++){
        cout<<allKeypoints[i].size()<<endl;
    }
    //层次越高，特征点应该更少一些才是？
    cv::Mat img_1, img_2, img_3, img_0;
    cv::drawKeypoints(img, allKeypoints[0], img_0);
    cv::drawKeypoints(img, allKeypoints[1], img_1);
    cv::drawKeypoints(img, allKeypoints[2], img_2);
    cv::drawKeypoints(img, allKeypoints[3], img_3);

    cv::imshow("0", img_0);
    cv::imshow("1", img_1);
    cv::imshow("2", img_2);
    cv::imshow("3", img_3);

    cv::waitKey(0);*/
    return 0;
}