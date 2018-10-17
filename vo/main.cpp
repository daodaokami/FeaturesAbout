//
// Created by lut on 18-10-12.
//

#include "common_include.h"
#include <vector>
#include <detector.h>
#include <detector_fast.h>
#include <detector_orb.h>
#include <detector_sift.h>
#include <opencv2/features2d.hpp>
#include <descriptor.h>
#include <desc_b256.h>
#include <chrono>

int main(){
    int num = 1;
    int on = num << 1;
    cout<<num<<", "<<on<<endl;//如果直接<<那么是赋值，当做二进制运算符则相当于×2
    cv::Mat uchar_mat(3, 3, CV_8UC1);
    cv::Mat float_mat(3, 3, CV_32FC1);
    uchar_mat.at<unsigned char>(0, 0) = 1;uchar_mat.at<unsigned char>(0, 1) = 1;uchar_mat.at<unsigned char>(0, 2) = 1;
    uchar_mat.at<unsigned char>(1, 0) = 2;uchar_mat.at<unsigned char>(1, 1) = 2;uchar_mat.at<unsigned char>(1, 2) = 2;
    uchar_mat.at<unsigned char>(2, 0) = 3;uchar_mat.at<unsigned char>(2, 1) = 3;uchar_mat.at<unsigned char>(2, 2) = 3;
    cout<<uchar_mat<<endl;
    uchar_mat.convertTo(float_mat, CV_32FC1);
    cout<<float_mat<<endl;
    //对图像进行上采样
    float scale = 2;
    cv::Size sz(cvRound((float)float_mat.cols*scale), cvRound((float)float_mat.rows*scale));
    cv::Mat temp;
    cv::resize(float_mat, temp, sz);
    cout<<"up sample \n"<<temp<<endl;
    //调bug没有准确的定位好特征点，计算的位置产生了偏移
    cv::Mat img;
    string filepath = "./data/000001.png";
    img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
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
    delete detector;

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