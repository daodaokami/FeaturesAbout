//
// Created by lut on 18-10-20.
//
#include <opencv2/features2d.hpp>
#include <fstream>
#include "gauss_blur.h"
#include "accum_conv.h"
#include "common_include.h"
using namespace std;

int main(){
    vector<cv::KeyPoint> kps;
    kps.reserve(1706);
    float x, y;
    ifstream infile("./data/positions.txt");
    if(!infile.is_open()) {
        cerr << "not open" << endl;
        return -1;
    }
    for(int i=0; i<1706; i++){
        infile>>x>>y;
        cv::KeyPoint kp;
        kp.pt.x = x;
        kp.pt.y = y;
        kps.push_back(kp);
    }
    cv::Mat img = cv::imread("./data/example.jpeg", cv::IMREAD_GRAYSCALE);
    cv::Mat out;
    cv::drawKeypoints(img, kps, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    return 0;
    /*int len = 10;
    cv::Mat fmat(len, len, CV_32FC1);
    for(int i=0; i<len*len; i++){
        fmat.at<float>(i/len, i%len) = (i+1);
    }
    cout<<"fmat :\n"<<fmat<<endl;
    cv::Mat res(len, len, CV_32FC1);
    float sigma = 1.5;
    cv::Size sz(ceil(sigma*2.884)*2+1, ceil(sigma*2.884)*2+1);
    cv::GaussianBlur(fmat, res, sz, sigma, sigma, cv::BORDER_REPLICATE);
    cout<<"res blur: \n"<<res<<endl;

    cv::Mat half;
    cv::Mat half2;
    cv::Size sz1(fmat.cols/2, fmat.rows/2);
    //利用gauss 函数来进行大小重置
    cv::resize(fmat, half, sz1, 0, 0, cv::INTER_LINEAR_EXACT);
    cv::pyrDown(fmat, half2, sz1);
    cout<<"resize \n"<<half<<endl;
    cout<<"resize2 \n"<<half<<endl;
    cv::Mat half3;
    suo15features::Gauss_Blur GB;
    half3 = GB.rescale_half_size_gaussian(fmat);
    cout<<"resize3 \n"<<half3<<endl;*/
    return 0;
}