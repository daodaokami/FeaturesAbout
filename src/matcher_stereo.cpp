//
// Created by lut on 18-11-7.
//

#include "matcher_stereo.h"
#include "algorithm"

namespace suo15features{
    void Matcher_stereo::createRowIndexes() {
        if (!this->vRowIndexes.empty())
            this->vRowIndexes.clear();
        this->vRowIndexes.resize(img_sz.height);//获得图像的高
        for(int i=0; i<img_sz.height; i++){
            vRowIndexes[i].reserve(200);
        }

        const int Nr = right_keypoints.size();
        for(int iR = 0; iR < Nr; iR ++){
            const cv::KeyPoint &kp = (right_keypoints)[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f*(mvScaleFactors)[(right_keypoints)[iR].octave];
            const int maxr = ceil(kpY+r);
            const int minr = floor(kpY-r);

            for(int yi = minr; yi <= maxr; yi++){
                vRowIndexes[yi].push_back(iR);
            }
        }
        /*
         * 创建好了索引之后,进行左右特征点的匹配
         * */
    }
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void Matcher_stereo::ComputeStereoMatches() {
        //主要获取左视图的深度信息
        int N = left_keypoints.size();
        mvuRight = std::vector<float>(N, -1.0f);
        mvDepth = std::vector<float>(N, -1.0f);

        const int thOrbDist = (options.TH_LOW + options.HISTO_LENGTH)/2;

        const float minZ = mb;
        const float minD = 0;
        const float maxD = 50;//mbf/minZ;

        std::vector<std::pair<int, int>> vDistIdx;
        vDistIdx.reserve(N);//左边的特征点对应的最近的右视图的点的集合
        this->vMatches.resize(N, -1);
        this->vDistance.resize(N, 255);
        for(int iL=0; iL < N; iL++){
            const cv::KeyPoint &kpL = (left_keypoints)[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const std::vector<size_t>& vCandidates = this->vRowIndexes[vL];
            //搜索范围,空间
            if(vCandidates.empty())
                continue;
            const float minU = uL - maxD;
            const float maxU = uL - minD;
            /*
             * 最小的深度是基线的长度
             * uL - maxD是最小可能的uR的位置
             * uL - minD是最大的uR的位置,但是不可能会大于uL
             * */
            if(maxU < 0)
                continue;

            int bestDist = options.TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat& dL = left_descriptors.row(iL);
            for(size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = (right_keypoints)[iR];
                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;
                //kpR的层次比当前特征点的层次多余两层的话,匹配不够准确,需要剔除
                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = right_descriptors.row(iR);
                    const int dist = DescriptorDistance(dL, dR);
                    /**
                     * change my distance calculate algorithom
                     * */
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                        this->vMatches[iL] = iR;
                        this->vDistance[iL] = dist;
                    }
                }
            }//这里就已经找到了相关的右边点的索引
            if(bestDist < thOrbDist){
                const float uR0 = (right_keypoints)[bestIdxR].pt.x;
                const float scaleFactor = 1.0f/(mvScaleFactors)[kpL.octave];
                const float scaleduL = round(kpL.pt.x*scaleFactor);
                const float scaledvL = round(kpL.pt.y*scaleFactor);
                const float scaleduR0 = round(uR0*scaleFactor);
                /*
                 * the first get the scale uR0, 右边匹配点的尺度下的x坐标值
                 * also used the imagePyramid to repositions the matched points
                 * */
                const int w = 5;
                //获取当前尺度下图像金字塔的范围
                cv::Mat IL = (left_imagePyramid)[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
                IL.convertTo(IL,CV_32F);
                IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
                //处理一下减去中间像素,得到去中心值的IL
                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                std::vector<float> vDists;
                vDists.resize(2*L+1);//11位的空间

                const float iniu = scaleduR0+L-w;
                const float endu = scaleduR0+L+w+1;
                if(iniu<0 || endu >= (right_imagePyramid)[kpL.octave].cols)
                    continue;

                for(int incR=-L; incR<=+L; incR++)
                {
                    cv::Mat IR = (right_imagePyramid)[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                    IR.convertTo(IR,CV_32F);
                    IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    float dist = cv::norm(IL,IR,cv::NORM_L1);
                    if(dist<bestDist)
                    {
                        bestDist =  dist;
                        bestincR = incR;
                    }

                    vDists[L+incR] = dist;
                }
                //在5*5的区间范围内光度差最小的点
                if(bestincR==-L || bestincR==L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L+bestincR-1];
                const float dist2 = vDists[L+bestincR];
                const float dist3 = vDists[L+bestincR+1];
                //在小范围内进行更加准确的定位与追踪
                const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
                //并且在best的incR周围找到亚像素精度的点
                if(deltaR<-1 || deltaR>1)
                    continue;
                // Re-scaled coordinate
                float bestuR = (mvScaleFactors)[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

                float disparity = (uL-bestuR);

                if(disparity>=minD && disparity<maxD)
                {
                    if(disparity<=0)
                    {
                        disparity=0.01;
                        bestuR = uL-0.01;
                    }
                    mvDepth[iL]=mbf/disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(std::pair<int,int>(bestDist,iL));
                }
            }
        }
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size()/2].first;
        const float thDist = 1.5f*1.4f*median;

        for(int i=vDistIdx.size()-1; i>=0; i--) {
            if(vDistIdx[i].first < thDist)
                break;
            else{
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }

    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        //计算距离
        return dist;
    }
}