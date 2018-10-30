//
// Created by lut on 18-10-26.
//

#include <iostream>
#include <detector_sift.h>
#include <desc_s128.h>
#include <sift_keypoint.h>
#include <matcher.h>
#include "common_include.h"

using namespace std;

void feature_set_matching(const cv::Mat& image1, const cv::Mat& image2);

int main(int argc, char* argv[]){
    if(argc < 3){
        cerr<<"syntax: "<<argv[0] << " image1 image2 "<<endl;
        return 1;
    }
    cout<< (1 << 0)<<endl;
    cout<< (1 << 1)<<endl;

    suo15features::Matcher_options sift_matching_opts;
    sift_matching_opts.lowe_ratio_threshold = 0.8f;
    sift_matching_opts.descriptor_length = 128;
    sift_matching_opts.distance_threshold = std::numeric_limits<float>::max();


    suo15features::Descriptors sift_descr1, sift_descr2;


    return 0;
}