//
// Created by lut on 18-10-26.
//

#include <iostream>
#include <detector_sift.h>
#include <desc_s128.h>
#include <sift_keypoint.h>
#include <matcher.h>
#include <feature_set.h>
#include <gms_matcher.h>
#include <opencv2/features2d.hpp>
#include <matcher_knn.h>
#include <chrono>
#include <matcher_stereo.h>
#include "visualize.h"
#include "common_include.h"

using namespace std;
void showWidthImgMatch(string winname , const cv::Mat& image_1, const vector<cv::KeyPoint>& keypoints_1,
                       const cv::Mat& image_2, const vector<cv::KeyPoint>& keypoints_2,
                       const vector<cv::DMatch>& matches);

void runGMSFilter(Mat& img1, vector<cv::KeyPoint>& kp1,
                  Mat& desc1,
                  Mat& img2, vector<cv::KeyPoint>& kp2,
                  Mat& desc2,
                  vector<DMatch>& matches_all, vector<DMatch>& matches_gms);
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
    suo15features::ORB_options orb_options(31, 15, 19, 3000, 1.2, 4, 20, 7);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    suo15features::Detector<cv::KeyPoint>* detector_orb = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints = detector_orb->ExtractorKeyPoints(img_1);
    suo15features::Descry<cv::KeyPoint>* orb_desc = new suo15features::Desc_b256(detector_orb);
    cv::Mat desc = orb_desc->ComputeDescriptor(img_1, keypoints);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout<<"detector time out "<< elapsed_seconds.count()<<"s."<<endl;
    //cout<<"desc1 row \n"<<desc.size<<"\n"<<desc<<endl;
    cv::Mat km1;
    cv::drawKeypoints(img_1, keypoints, km1);
    cv::imshow("km1", km1);
    cv::waitKey(0);

    suo15features::Detector<cv::KeyPoint>* detector_orb2 = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints2 = detector_orb2->ExtractorKeyPoints(img_2);
    suo15features::Descry<cv::KeyPoint>* orb_desc2 = new suo15features::Desc_b256(detector_orb2);
    cv::Mat desc2 = orb_desc2->ComputeDescriptor(img_2, keypoints2);

    cv::Mat km2;
    cv::drawKeypoints(img_2, keypoints2, km2);
    cv::imshow("km2", km2);
    cv::waitKey(0);

    std::chrono::system_clock::time_point start_bf = std::chrono::system_clock::now();
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(desc, desc2, matches);
    std::chrono::system_clock::time_point end_bf = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_bf = end_bf - start_bf;
    cout<<"bfmatches size is "<<matches.size()<<endl;
    cout<<"bfmatcher time out "<< elapsed_seconds_bf.count()<<"s."<<endl;
    showWidthImgMatch("bf_match", img_1, keypoints, img_2, keypoints2, matches);
    cv::waitKey(0);
    //在bfmatches 之后,选择进行GMS的筛选
    /*
     *
     *
     * GMS filter the params is
     *
     * */
    vector<DMatch> matches_gms;
    std::chrono::system_clock::time_point start_gms = std::chrono::system_clock::now();
    runGMSFilter(img_1, keypoints, desc, img_2, keypoints2, desc2, matches, matches_gms);
    std::chrono::system_clock::time_point end_gms = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_gms = end_gms - start_gms;
    cout<<"gms matches size is "<<matches_gms.size()<<endl;
    cout<<"gms time out "<<elapsed_seconds_gms.count()<<"s."<<endl;
    showWidthImgMatch("gms_match", img_1, keypoints, img_2, keypoints2, matches_gms);
    cv::waitKey(0);

    /****              !!!!!             ****/
    //如何给出,高精度的match
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

    cout<<"knn_goodMatches size is "<<knn_goodMatches.size()<<endl;
    /*cv::Mat img_knngoodMatch;
    cv::drawMatches(img_1, keypoints, img_2, keypoints2, knn_goodMatches, img_knngoodMatch);
    cv::imshow("knn_goodMatch", img_knngoodMatch);
    cv::waitKey(0);*/

    showWidthImgMatch("wknn_goodmatch", img_1, keypoints, img_2, keypoints2, knn_goodMatches);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //如果用暴力匹配呢??
    /*
     * here test the matcher_stereo
     *
     *
    suo15features::Stereo_options options;
    cv::Size sz(img_1.size());
    cout<<"sz.height "<<sz.height<<", sz.width "<<sz.width<<endl;
    suo15features::Matcher_stereo matcher_stereo(options, sz, keypoints, keypoints2,
        desc, desc2, 1, 500, detector_orb->GetScaleFactors());
    matcher_stereo.setImagePyramid(detector_orb->GetImagePyramid(), detector_orb2->GetImagePyramid());
    matcher_stereo.createRowIndexes();
    matcher_stereo.ComputeStereoMatches();
    vector<int> vMatches = matcher_stereo.GetMatches();
    vector<int> vDistance = matcher_stereo.GetDistance();
    vector<cv::DMatch> vDMatches;
    vDMatches.reserve(vMatches.size());
    for(int i=0; i<vMatches.size(); i++){
        if(vMatches[i]!=-1){
            cv::DMatch dmatch;
            dmatch.queryIdx = i;
            dmatch.trainIdx = vMatches[i];
            dmatch.distance = vDistance[i];
            vDMatches.push_back(dmatch);
        }
    }

    cv::Mat out2;
    cv::drawMatches(img_1, keypoints, img_2, keypoints2, vDMatches, out2);
    cv::imshow("out2", out2);
    showWidthImgMatch("out3", img_1, keypoints, img_2, keypoints2, vDMatches);
    cv::waitKey(0);
    //双目匹配,恢复视差也完成了,接下来可以进行双目的tarcking
    //改善收匹配的数量有所增加,但是有许多的错误匹配!!!要进行二次修改了
    cout<<knn_goodMatches.size()<<", improve "<<vDMatches.size()<<endl;*/
    return 0;
}

void GmsMatch(Mat& img1, vector<cv::KeyPoint>& kp1,
              Mat& desc1,
              Mat& img2, vector<cv::KeyPoint>& kp2,
              Mat& desc2, vector<DMatch>& matches_all, vector<DMatch>& matches_gms);
void runGMSFilter(Mat& img1, vector<cv::KeyPoint>& kp1,
                  Mat& desc1,
                  Mat& img2, vector<cv::KeyPoint>& kp2,
                  Mat& desc2, vector<DMatch>& matches_all,
                  vector<DMatch>& matches_gms){
    GmsMatch(img1, kp1, desc1,
             img2, kp2, desc2, matches_all, matches_gms);
}

void GmsMatch(Mat& img1, vector<cv::KeyPoint>& kp1,
                Mat& desc1,
                Mat& img2, vector<cv::KeyPoint>& kp2,
                Mat& desc2, vector<DMatch>& matches_all, vector<DMatch>& matches_gms){
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    // collect matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }

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
