//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DETECTOR_ORB_H
#define LUT15VO_DETECTOR_ORB_H

#include "../include/detector.h"
#include "../include/detector_orb.h"

namespace suo15features {
    struct ORB_config{
        int _patch_size;
        int _half_path_size;
        int _edge_threshold;
        int _nfeatures;//预估计的特征的数量
        float _scaleFactor;
        int _nlevels;
        int _iniThFAST;//可接受的特征点的阈值
        int _minThFAST;//最小的可接受特征点的阈值
        ORB_config(){}
        ORB_config(int patch_size, int half_path_size, int edge_threshold,
                   int nfeatures, float scaleFactor, int nlevels,
                   int iniThFAST, int minThFAST): _patch_size(patch_size), _half_path_size(half_path_size),
                                                  _edge_threshold(edge_threshold), _nfeatures(nfeatures),
                                                  _scaleFactor(scaleFactor), _nlevels(nlevels),
                                                  _iniThFAST(iniThFAST), _minThFAST(minThFAST)
        {}
        void SetConfig(ORB_config config)
        {
            _patch_size = config._patch_size;
            _half_path_size = config._half_path_size;
            _edge_threshold = config._edge_threshold;
            _nfeatures = config._nfeatures;
            _scaleFactor = config._scaleFactor;
            _nlevels = config._nlevels;
            _iniThFAST = config._iniThFAST;
            _minThFAST = config._minThFAST;
        }
    };

    class Detector_orb :public Detector{
        //多了更多的层次的信息，旋转信息等等
        //提取ORB特征，还要额外加上旋转，金字塔等信息！！！
    private:
        ORB_config _config;
    protected:
        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;
        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        vector<cv::Point> pattern;
        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(vector<vector<cv::KeyPoint>>& allKeypoints);

    public:
        enum{HARRIS_SCORE = 0, FAST_SCORE = 1};
        Detector_orb();
        Detector_orb(ORB_config config);

        virtual vector<cv::KeyPoint> ExtractorKeyPoints(const cv::Mat& ori_img);

        vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys,
                                               const int &minX, const int &maxX, const int &minY,
                                               const int &maxY, const int &nFeatures, const int &level);


        vector<cv::Mat> mvImagePyramid;
        vector<int> mnkeypointsLevels;

        virtual vector<cv::Mat> GetImagePyramid(){
            return mvImagePyramid;
        }

        virtual vector<int> GetKeypointsLevels(){
            return mnkeypointsLevels;
        }

        int inline GetLevels(){
            return _config._nlevels;
        }

        float inline GetScaleFactor(){
            return _config._scaleFactor;
        }

        vector<float> inline GetScaleFactors(){
            return mvScaleFactor;
        }

        vector<float> inline GetInverseScaleFactors(){
            return mvInvScaleFactor;
        }

        vector<float> inline GetScaleSigmaSquares(){
            return mvLevelSigma2;
        }

        vector<float> inline GetInverseScaleSigmaSquares(){
            return mvInvLevelSigma2;
        }
    };
}

#endif //LUT15VO_ORB_H
