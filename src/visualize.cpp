//
// Created by lut on 18-10-31.
//

#include <opencv2/features2d.hpp>
#include <matcher.h>
#include "visualize.h"

namespace suo15features {
    void ShowImgWithKeypoints(const string winname, const cv::Mat &img, const vector<cv::Point2f> &positions) {
        vector<cv::KeyPoint> keypoints;
        keypoints.resize(positions.size());
        for (size_t i = 0; i < positions.size(); i++) {
            keypoints[i].pt = positions[i];
        }
        cv::Mat out;
        cv::drawKeypoints(img, keypoints, out);
        cv::imshow(winname, out);
    }

    void ShowImgWithMatches(const string winname, const cv::Mat& img,
                            const vector<cv::Point2f>& positions_1,
                            const vector<cv::Point2f>& positions_2,
                            const Result& result){
        vector<cv::KeyPoint> keypoints_1(positions_1.size());
        vector<cv::KeyPoint> keypoints_2(positions_2.size());

        for(int i=0; i<keypoints_1.size(); ++i){
            keypoints_1[i].pt = positions_1[i];
        }

        for(int i=0; i<keypoints_2.size(); ++i){
            keypoints_2[i].pt = positions_2[i];
        }

        vector<cv::DMatch> matches;
        //在maches_1_2中有的匹配点与matches_2_1有的一样的匹配点，进行特征匹配
        for(int i=0; i<result.matches_1_2.size(); i++){

        }

    }
}