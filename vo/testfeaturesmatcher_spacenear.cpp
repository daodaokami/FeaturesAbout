//
// Created by lut on 18-12-25.
//

#include <detector_orb.h>
#include <descry.h>
#include <desc_b256.h>
#include <datas_map.h>
#include <chrono>
#include "spacenear_matcher.h"
#include "spatical_subdivision.h"
#include "common_include.h"
#include "gms_matcher.h"

using namespace std;

void GmsMatch(Mat &img1, Mat &img2, vector<cv::KeyPoint>& kps1, vector<cv::KeyPoint>& kps2,
              vector<cv::DMatch>& all_matches);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void runImagePair(cv::Mat& img1, cv::Mat& img2, vector<cv::KeyPoint>& kps1, vector<cv::KeyPoint>& kps2,
                  vector<cv::DMatch>& all_matches) {
    //time cost Most in the detector and match
    //GMS just cost 1~2 ms,it is so fast and good to do with slam
    //how it performance in the uniform features
    GmsMatch(img1, img2, kps1, kps2, all_matches);
}

int main(int argc, char* argv[]){
    if(argc < 3) {
        cerr<<"do not have enough params (please input two img path)!"<<endl;
        return -1;
    }

    string img_path_1 = argv[1];
    string img_path_2 = argv[2];

    cv::Mat img_1 = cv::imread(img_path_1, cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(img_path_2, cv::IMREAD_GRAYSCALE);

    /*vector<cv::KeyPoint> keypoints, keypoints2;
    cv::Mat desc, desc2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    orb->setFastThreshold(5);
    orb->detectAndCompute(img_1, cv::Mat(), keypoints, desc);
    orb->detectAndCompute(img_2, cv::Mat(), keypoints2, desc2);*/
    suo15features::ORB_options orb_options(21, 15, 11, 2000, 1.1, 4, 11, 7);
    suo15features::Detector<cv::KeyPoint>* detector_orb = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints = detector_orb->ExtractorKeyPoints(img_1);
    suo15features::Descry<cv::KeyPoint>* orb_desc = new suo15features::Desc_b256(detector_orb);
    cv::Mat desc = orb_desc->ComputeDescriptor(img_1, keypoints);

    suo15features::Detector<cv::KeyPoint>* detector_orb2 = new suo15features::Detector_orb(orb_options);
    vector<cv::KeyPoint> keypoints2 = detector_orb2->ExtractorKeyPoints(img_2);
    suo15features::Descry<cv::KeyPoint>* orb_desc2 = new suo15features::Desc_b256(detector_orb2);
    cv::Mat desc2 = orb_desc2->ComputeDescriptor(img_2, keypoints2);

    //得到了特征点与描述子
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches_all;
    matcher.match(desc, desc2, matches_all);
    cout<<"matches all is "<<matches_all.size()<<endl;

    Spatical_Subdivision sp_subdiv(img_1.size(), 80);
    //一项一项的测试
    /**
     * 特征点所在区域划分
     *
     * */
    sp_subdiv.SplitPoints2Index(img_1.size(), keypoints);
    std::vector<std::vector<int>> vIndexes = sp_subdiv.GetvIndexes();
    int count = 0;
    for(int i=0; i<vIndexes.size(); i++){
        cout<<"i "<<i<<" "<<vIndexes[i].size()<<endl;
        if(vIndexes[i].size()!=0)
            count++;
    }
    cout<<"img1 not zero Area "<<count<<endl;

    Spatical_Subdivision sp_subdiv2(img_2.size(), 80);
    sp_subdiv2.SplitPoints2Index(img_2.size(), keypoints2);
    std::vector<std::vector<int>> vIndexes2 = sp_subdiv2.GetvIndexes();
    int count2 = 0;
    for(int i=0; i<vIndexes2.size(); i++){
        cout<<"i "<<i<<" "<<vIndexes2[i].size()<<endl;
        if(vIndexes[i].size()!=0)
            count2++;
    }
    cout<<"img2 not zero Area "<<count2<<endl;

    GridNet gridNet, gridNet2;
    sp_subdiv.ComputeGirdAndNeighbors(img_1.size(), keypoints, desc, gridNet);
    sp_subdiv.ComputeGirdAndNeighbors(img_2.size(), keypoints2, desc2, gridNet2);
    cout<<"it is over!"<<endl;

    cv::Mat img1_kps, img2_kps;
    cv::drawKeypoints(img_1, keypoints, img1_kps);
    cv::drawKeypoints(img_2, keypoints2, img2_kps);
    cv::imshow("img1_kps", img1_kps);
    cv::imshow("img2_kps", img2_kps);
    //cv::waitKey(0);

    SpaceNear_Matcher spacenear_matcher(&gridNet, &gridNet2);
    vector<cv::DMatch> all_matches;
    vector<cv::KeyPoint> keyps_1, keyps_2;
    spacenear_matcher.SpaceNearMatcher(keyps_1, keyps_2, all_matches);
    cout<<"k1 - "<<keyps_1.size()<<" k2 - "<<keyps_2.size()<<" m - "<<all_matches.size()<<endl;

    int preSize = keypoints2.size();
    KeyPoints_Map kps_map(preSize);
    /*注意这两个功能必须得是同时使用的,一定对相同的数据进行操作*/
    kps_map.CreateUnorder_Map_KpIndex_vecIndex_IndexKP(keypoints2);
    kps_map.CreateUnorder_Map_OuterIndex_InnerIndex(keyps_2);
    map<int, int>* indexMap = kps_map.GetIndexMap();
    for(int vm_index=0; vm_index<all_matches.size(); vm_index++){
        int outer_index = all_matches[vm_index].trainIdx;
        if(indexMap->find(outer_index) == indexMap->end()){
            cerr<<"some points can not find!"<<endl;
        }
        else{
            all_matches[vm_index].trainIdx = (*indexMap)[outer_index];
        }
    }
    /*
     *
     * !!!!!!!!!!!!!                   !!!!!!!!!!!!!!!
     *
     * */
    cout<<"###############################"<<endl;

    for(int i=0; i<all_matches.size(); i++){
        cout<<all_matches[i].trainIdx<<"     "<<all_matches[i].queryIdx<<endl;
    }
    cout<<keyps_1.size()<<endl;
    cv::Mat image;
    cv::drawMatches(img_1, keyps_1, img_2, keypoints2, all_matches, image);
    cv::imshow("show", image);
    runImagePair(img_1, img_2, keyps_1, keypoints2, all_matches);
    cv::destroyAllWindows();

}


void GmsMatch(Mat &img1, Mat &img2, vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2,
              vector<cv::DMatch>& matches_all) {

    vector<DMatch> matches_gms;

    std::chrono::system_clock::time_point start_gms = std::chrono::system_clock::now();
    // GMS filter
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
    std::chrono::system_clock::time_point end_gms = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_gms = end_gms - start_gms;

    //just one or two sub areas is too small to do the matches
    cout << "Get total " << num_inliers << " matches." << endl;
    cout<<"GMS time out "<< elapsed_seconds_gms.count()<<"s."<<endl;
    // draw matching
    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("show", show);
    waitKey();
}

void ShowRotationMatchesImage(cv::Mat& img_1, vector<cv::KeyPoint>& kps_1,
                              cv::Mat& img_2, vector<cv::KeyPoint>& kps_2,
                              vector<cv::DMatch>& matches);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));

    Mat three_channel_1 = Mat::zeros(src1.rows, src1.cols, CV_8UC3);
    vector<Mat> channels_1;
    for (int i=0;i<3;i++)
    {
        channels_1.push_back(src1);
    }
    merge(channels_1,three_channel_1);

    Mat three_channel_2 = Mat::zeros(src2.rows, src2.cols, CV_8UC3);
    vector<Mat> channels_2;
    for (int i=0;i<3;i++)
    {
        channels_2.push_back(src2);
    }
    merge(channels_2,three_channel_2);

    three_channel_1.copyTo(output(Rect(0, 0, three_channel_1.cols, three_channel_1.rows)));
    three_channel_2.copyTo(output(Rect(three_channel_1.cols, 0, three_channel_2.cols, three_channel_2.rows)));

    if (type == 1)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            line(output, left, right, Scalar(0, 255, 255));
        }
    }
    else if (type == 2)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            line(output, left, right, Scalar(255, 0, 0));
        }

        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)three_channel_1.cols, 0.f));
            circle(output, left, 1, Scalar(0, 255, 255), 2);
            circle(output, right, 1, Scalar(0, 255, 0), 2);
        }
    }
    ShowRotationMatchesImage(src1, kpt1, src2, kpt2, inlier);
    return output;
}


void ShowRotationMatchesImage(cv::Mat& img_1, vector<cv::KeyPoint>& kps_1,
                              cv::Mat& img_2, vector<cv::KeyPoint>& kps_2,
                              vector<cv::DMatch>& matches){
    //将img逆时针旋转90°,特征点的x,y交换位置
    cv::Mat r90_img_1, r90_img_2;
    transpose(img_1, r90_img_1);
    transpose(img_2, r90_img_2);

    vector<cv::KeyPoint> transpose_kps_1, transpose_kps_2;
    transpose_kps_1.resize(kps_1.size());
    transpose_kps_2.resize(kps_2.size());
    for(int i=0; i<kps_1.size(); i++){
        transpose_kps_1[i].pt.x = kps_1[i].pt.y;
        transpose_kps_1[i].pt.y = kps_1[i].pt.x;
    }

    for(int i=0; i<kps_2.size(); i++){
        transpose_kps_2[i].pt.x = kps_2[i].pt.y;
        transpose_kps_2[i].pt.y = kps_2[i].pt.x;
    }
    cv::Mat out;
    cv::drawMatches(r90_img_1, transpose_kps_1, r90_img_2, transpose_kps_2, matches, out);
    transpose(out, out);
    cv::imshow("out", out);
    cv::waitKey(0);
}