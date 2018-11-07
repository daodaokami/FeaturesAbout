//
// Created by lut on 18-10-12.
//

#include "common_include.h"
#include <detector.h>
#include <detector_fast.h>
#include <detector_orb.h>
#include "sift_keypoint.h"
#include <detector_sift.h>
#include <opencv2/features2d.hpp>
#include <descry.h>
#include <desc_s128.h>
#include <chrono>
#include <desc_b256.h>

bool sift_compare(suo15features::Sift_KeyPoint& kp1, suo15features::Sift_KeyPoint& kp2){
    return kp1.scale > kp2.scale;
}

bool sift_compare_x(suo15features::Sift_KeyPoint& kp1, suo15features::Sift_KeyPoint& kp2){
    return kp1.pt.x < kp2.pt.x;
}
int main(){
    /*cv::Mat img_example = cv::imread("./data/example.jpeg", cv::IMREAD_ANYCOLOR);
    suo15features::Detector_sift* Ptr = new suo15features::Detector_sift(suo15features::SIFT_options());
    //注意这里返回的值的类型还要修改一波！！！最好还是用模板类型，但是模板类型不能用虚函数
    vector<suo15features::Sift_KeyPoint> sift_keypoints = Ptr->ExtractorKeyPoints(img_example);
    vector<cv::KeyPoint> vkps(sift_keypoints.size());
    for(size_t i=0; i<sift_keypoints.size(); i++){
        cout<<sift_keypoints[i].pt.x<<" "<<sift_keypoints[i].pt.y<<" "<<sift_keypoints[i].sample<<endl;
        vkps[i].pt = sift_keypoints[i].pt;
    }
    cv::Mat out;
    cv::drawKeypoints(img_example, vkps, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    suo15features::S128_options desc_s128_options;
    suo15features::Descry<suo15features::Sift_KeyPoint>* descry = new suo15features::Desc_s128(desc_s128_options);
    cv::Mat descriptors = descry->ComputeDescriptor(img_example, sift_keypoints);
    cout<<"new_keypoints "<<sift_keypoints.size()<<endl;
    cout<<"descriptors size "<<descriptors.size()<<endl;

    sort(sift_keypoints.begin(), sift_keypoints.end(), sift_compare);

    vector<cv::KeyPoint> show_kps;
    for(size_t i=0; i<(sift_keypoints.size()); i++)
    {
        cv::KeyPoint kp;
        kp.pt = sift_keypoints[i].pt;
        show_kps.push_back(kp);
        cout<<kp.pt<<" "<<sift_keypoints[i].scale<<endl;
    }
    cv::Mat out2;
    cv::drawKeypoints(img_example, show_kps, out2);
    cv::imshow("out2", out2);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;*/

    /*cv::Mat img_example = cv::imread("./data/example.jpeg", cv::IMREAD_ANYCOLOR);
    suo15features::Detector_sift::SIFT_Options sift_options;
    suo15features::Detector<suo15features::Sift_KeyPoint>* detector = new suo15features::Detector_sift(sift_options);
    vector<suo15features::Sift_KeyPoint> SKP = detector->ExtractorKeyPoints(img_example);
    vector<cv::KeyPoint> kps;
    kps.resize(SKP.size());
    for(size_t i=0; i<SKP.size(); i++){
        kps[i].pt = SKP[i].pt;
    }
    cv::Mat out;
    cv::drawKeypoints(img_example, kps, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    return 0;*/

   /* cv::Mat img;
    string filepath = "./data/000001.png";

    img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

    suo15features::Detector<cv::KeyPoint> *detector;
    detector = new suo15features::Detector_fast();
    vector<cv::KeyPoint> keypoints_fast = detector->ExtractorKeyPoints(img);
    cout<<keypoints_fast.size()<<endl;
    cv::Mat out;
    cv::drawKeypoints(img, keypoints_fast, out);
    cv::imshow("keypoints", out);
    cv::waitKey(0);
    delete detector;
    return 0;*/
/*

    suo15features::ORB_config default_config(31, 15, 19, 1000, 1.2, 4, 20, 7);
    detector = new suo15features::Detector_orb(default_config);
    vector<int> mnkeypointsLevels = detector->GetKeypointsLevels();

    suo15features::Descry* descry;
    suo15features::Detector_orb detector_orb;
    //descriptor = new suo15features::Desc_b256();
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    vector<cv::KeyPoint> keypoints_orb = detector->ExtractorKeyPoints(img);

    descry = new suo15features::Desc_b256(detector);
    cv::Mat descs = descry->ComputeDescriptor(img, keypoints_orb);

    mnkeypointsLevels.clear();
    mnkeypointsLevels = detector->GetKeypointsLevels();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;
    delete detector;
*/
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

    cv::waitKey(0);
    return 0;*/
}