//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_DETECTOR_FAST_H
#define LUT15VO_DETECTOR_FAST_H

#include "detector.h"
#include "detector_fast.h"

namespace suo15features {

    struct Fast_config{
        int _patch_size;
        int _half_path_size;
        int _edge_threshold;
        int _nfeatures;//预估计的特征的数量
        int _iniThFAST;//可接受的特征点的阈值
        int _minThFAST;//最小的可接受特征点的阈值
        Fast_config(){}
        Fast_config(int patch_size, int half_path_size, int edge_threshold,
            int nfeatures, int iniThFAST, int minThFAST): _patch_size(patch_size), _half_path_size(half_path_size),
                                                          _edge_threshold(edge_threshold), _nfeatures(nfeatures),
                                                          _iniThFAST(iniThFAST), _minThFAST(minThFAST)
        {}
        void SetConfig(Fast_config config)
        {
            _patch_size = config._patch_size;
            _half_path_size = config._half_path_size;
            _edge_threshold = config._edge_threshold;
            _nfeatures = config._nfeatures;
            _iniThFAST = config._iniThFAST;
            _minThFAST = config._minThFAST;
        }
        /*
         * init value is patch_size = 31,
         *               half_patch_size = 15,
         *               edge_threshold = 19,
         *               nfeatures = 2000,
         *               iniThFAST = 20,
         *               minThFAST = 7//当提取的fast特征点少的时候才使用minThFAST参数
         * */
    };

    class Detector_fast:public Detector<cv::KeyPoint>{
    private:
        Fast_config _config;
        vector<int> umax;
    public:
        Detector_fast();
        Detector_fast(Fast_config config);
        virtual vector<cv::KeyPoint> ExtractorKeyPoints(const cv::Mat& ori_img);

        vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys,
                          const int &minX, const int &maxX, const int &minY,
                          const int &maxY, const int &nFeatures);
    };
}
#endif //LUT15VO_FAST_H
