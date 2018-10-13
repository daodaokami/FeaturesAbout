//
// Created by lut on 18-10-12.
//

#include "common_include.h"
#include <vector>
#include <detector.h>
#include <detector_fast.h>
#include <opencv2/features2d.hpp>
#include <chrono>

int main(){

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

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    vector<cv::KeyPoint> keypoints = detector->ExtractorKeyPoints(img);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;

    cv::Mat img1;
    cv::drawKeypoints(img, keypoints, img1);
    //都还没有进行非极大抑制！！！
    cv::imshow("fast_img0", img1);
    cv::waitKey(0);
    return 0;
}