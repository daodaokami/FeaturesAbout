//
// Created by lut on 18-10-12.
//

#include <opencv2/features2d.hpp>
#include "common_include.h"
#include "../include/detector_fast.h"

namespace suo15features{
    Detector_fast::Detector_fast(){
        _config._patch_size = 31;
        _config._half_path_size = 15;
        _config._edge_threshold = 19;
        _config._nfeatures = 2000;//预估计的特征的数量
        _config._iniThFAST = 20;//可接受的特征点的阈值
        _config._minThFAST = 7;//最小的可接受特征点的阈值
    }

    Detector_fast::Detector_fast(Fast_config config){
        _config.SetConfig(config);
    }

    vector<cv::KeyPoint> Detector_fast::ExtractorKeyPoints(const cv::Mat& ori_img){

        vector<cv::KeyPoint> keypoints;
        keypoints.resize(_config._nfeatures);

        if(ori_img.empty()) {
            keypoints.resize(0);
            return keypoints;
        }

        cv::Mat image = ori_img.clone();
        assert(image.type() == CV_8UC1);
        //一般都是先进行特征点的提取--之后GaussBlur--再进行描述子的计算
        //cv::GaussianBlur(image, image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        const float W = 30;

        const int minBorderX = _config._edge_threshold - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = image.cols - _config._edge_threshold + 3;
        const int maxBorderY = image.rows - _config._edge_threshold + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(_config._nfeatures*10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = static_cast<int>(width / W);
        const int nRows = static_cast<int>(height/ W);
        const int wCell = static_cast<int>(ceil(width/nCols));
        const int hCell = static_cast<int>(ceil(height/nRows));

        for(int i=0; i < nRows; i++){
            const float iniY = minBorderY + i*hCell;
            float maxY = iniY + hCell + 6;
            if(iniY >= maxBorderY - 3)
                continue;
            if(maxY > maxBorderY)
                maxY = maxBorderY;
            for(int j=0; j<nCols; j++){
                const float iniX = minBorderX + j*wCell;
                float maxX = iniX + wCell + 6;
                if(iniX >= maxBorderX-6)
                    continue;
                if(maxX>maxBorderX)
                    maxX = maxBorderX;
                vector<cv::KeyPoint> vKeysCell;
                //确定Cell的位置
                cv::FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX),
                    vKeysCell, _config._iniThFAST, true);
                if(vKeysCell.empty()){
                    FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX),
                    vKeysCell, _config._minThFAST, true);
                }
                if(!vKeysCell.empty()){
                    for(vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
                            vit!=vKeysCell.end(); vit++){
                        (*vit).pt.x += j*wCell;
                        (*vit).pt.y += i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }//将vector key转换成结构的特征点存储，八叉树的结构
                }
            }
        }
        keypoints.reserve(_config._nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                minBorderY, maxBorderY, _config._nfeatures);
        const int scaledPatchSize = _config._patch_size;

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave = 0;
            keypoints[i].size = scaledPatchSize;
        }

        return keypoints;
    }

    vector<cv::KeyPoint> Detector_fast::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                                           const int &minX, const int &maxX, const int &minY,
                                                           const int &maxY, const int &nFeatures) {
        //计算有几个树的root节点，能够为
        const int nIni = static_cast<int>(
                round(static_cast<float>(maxX-minX)/(maxY-minY))//即是长和宽的比
        );
        //每块要在正方形的框内
        const float hX = static_cast<float>(maxX - minX)/nIni;
        list<ExtractorNode> lNodes;
        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++){
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
            ni.UR = cv::Point2i(hX * static_cast<float>(i+1), 0);
            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);

            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        for(size_t i=0; i<vToDistributeKeys.size(); i++){
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
        }

        //特征点划分到不同的extractor中，进行数据划分，如果有一块没有特征点，那么erase掉那一块
        list<ExtractorNode>::iterator lit = lNodes.begin();
        while(lit!=lNodes.end()){
            if(lit->vKeys.size() == 1){
                lit->bNoMore = true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        //假设只有lNodes只存了一个EtractorNode*
        bool bFinish = false;
        int iteration = 0;
        vector<pair<int, ExtractorNode*>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish){
            iteration++;
            int prevSize = lNodes.size();
            //prevSize = 1
            lit = lNodes.begin();
            int nToExpand = 0;
            vSizeAndPointerToNode.clear();

            while(lit != lNodes.end()) {
                if (lit->bNoMore) {//
                    lit++;
                    continue;
                } else {
                    ExtractorNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4);//把当前份割的四个
                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit = lNodes.erase(lit);
                    continue;
                }
            }
            //lNodes.size 不可能会大于nFeatures的
            if((int)lNodes.size() >= nFeatures||
                    (int)lNodes.size() == prevSize){
                bFinish = true;
            }else if(((int)lNodes.size() + nToExpand*3)>nFeatures){
                while(!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                        // Add childs if they contain points
                        if(n1.vKeys.size()>0)
                        {
                            lNodes.push_front(n1);
                            if(n1.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n2.vKeys.size()>0)
                        {
                            lNodes.push_front(n2);
                            if(n2.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n3.vKeys.size()>0)
                        {
                            lNodes.push_front(n3);
                            if(n3.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n4.vKeys.size()>0)
                        {
                            lNodes.push_front(n4);
                            if(n4.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if((int)lNodes.size()>=nFeatures)
                            break;
                    }
                    //这里进行初始的响应值的筛选，当提取的特征较多的时候
                    if((int)lNodes.size()>=nFeatures || (int)lNodes.size()==prevSize)
                        bFinish = true;
                }
            }
        }


        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nFeatures);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {//这是做非极大抑制吗？？
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].response>maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }
}