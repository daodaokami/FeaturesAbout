//
// Created by lut on 18-10-12.
//
#include <opencv2/features2d.hpp>
#include "../include/detector_orb.h"

namespace suo15features{
    //static ORB_config default_config(31, 15, 19, 1000, 0.8, 4, 20, 7);
    Detector_orb::Detector_orb(){}
    Detector_orb::Detector_orb(ORB_config config) :_config(config){
        mvScaleFactor.resize(_config._nlevels);
        mvLevelSigma2.resize(_config._nlevels);
        mnkeypointsLevels.resize(_config._nlevels);

        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;

        for(int i=1; i<_config._nlevels; i++){
            mvScaleFactor[i] = mvScaleFactor[i-1]*_config._scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(_config._nlevels);
        mvInvLevelSigma2.resize(_config._nlevels);
        for(int i=0; i<_config._nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(_config._nlevels);
        mnFeaturesPerLevel.resize(_config._nlevels);

        float factor = 1.0f/ _config._scaleFactor;
        float nDesiredFeaturesPerScale = _config._nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)_config._nlevels));

        int sumFeatures = 0;
        for(int level = 0; level < _config._nlevels-1; level++){
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[_config._nlevels-1] = std::max(_config._nfeatures-sumFeatures, 0);

        //const int npoints = 512;
        //const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
        //这个是描述子

        //Orientation
        umax.resize(_config._half_path_size + 1);
        int v, v0, vmax = cvFloor(_config._half_path_size*sqrt(2.f)/2 +1);
        int vmin = cvCeil(_config._half_path_size*sqrt(2.f)/2);
        const double hp2 = _config._half_path_size*_config._half_path_size;
        for(v = 0; v<= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 -v*v));

        for(v = _config._half_path_size, v0=0; v>=vmin; --v){
            while(umax[v0] == umax[v0+1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
        //首先配制参数！！
    }

    vector<cv::KeyPoint> Detector_orb::ExtractorKeyPoints(const cv::Mat &ori_img) {
        vector<cv::KeyPoint> keypoints;

        if(ori_img.empty())
            return keypoints;
        Mat image = ori_img.clone();
        assert(image.type() == CV_8UC1);

        ComputePyramid(image);

        vector<vector<cv::KeyPoint>> allKeypoints;

        ComputeKeyPointsOctTree(allKeypoints);
        //这里就提取好了特征点

        int nKeypoints = 0;
        for(int level = 0; level<_config._nlevels; ++level)
            nKeypoints += (int)(allKeypoints[level].size());

        keypoints.clear();
        keypoints.reserve(nKeypoints);

        int offset = 0;
        for(int level = 0; level < _config._nlevels; ++level){
            vector<cv::KeyPoint> &kps = allKeypoints[level];
            int nkeypointsLevel = (int)kps.size();
            mnkeypointsLevels[level] = nkeypointsLevel;
            if(nkeypointsLevel == 0)
                continue;

            offset += nkeypointsLevel;
            if(level!=0){
                float scale = mvScaleFactor[level];
                for(vector<cv::KeyPoint>::iterator itkp = kps.begin(),
                        kpend = kps.end(); itkp != kpend; ++itkp)
                    itkp->pt *= scale;
            }
            keypoints.insert(keypoints.end(), kps.begin(), kps.end());
        }
        return keypoints;
    }

    static float IC_Angle(const Mat& image, const ORB_config& config, cv::Point2f pt,  const vector<int> & u_max) {
        int m_01 = 0, m_10 = 0;

        const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -config._half_path_size; u <= config._half_path_size; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int) image.step1();
        for (int v = 1; v <= config._half_path_size; ++v) {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u) {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return cv::fastAtan2((float) m_01, (float) m_10);
    }

    static void computeOrientation(const Mat& image, const ORB_config& config, vector<cv::KeyPoint>& keypoints, const vector<int>& umax)
    {
        for (vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = IC_Angle(image, config, keypoint->pt, umax);
        }
    }

    vector<cv::KeyPoint> Detector_orb::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY,
                                                         const int &nFeatures, const int &level) {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

        const float hX = static_cast<float>(maxX-minX)/nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
            ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
            ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
            ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        //Associate points to childs
        for(size_t i=0;i<vToDistributeKeys.size();i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1)
            {
                lit->bNoMore=true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while(lit!=lNodes.end())
            {
                if(lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit=lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if((int)lNodes.size()>=nFeatures || (int)lNodes.size()==prevSize)
            {
                bFinish = true;
            }
            else if(((int)lNodes.size()+nToExpand*3)>nFeatures)
            {

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

                    if((int)lNodes.size()>=nFeatures || (int)lNodes.size()==prevSize)
                        bFinish = true;

                }
            }
        }

        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(_config._nfeatures);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
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

    void Detector_orb::ComputeKeyPointsOctTree(vector<vector<cv::KeyPoint>> &allKeypoints) {
        allKeypoints.resize(_config._nlevels);

        const float W = 30;

        for (int level = 0; level < _config._nlevels; ++level)
        {
            const int minBorderX = _config._edge_threshold-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-_config._edge_threshold+3;
            const int maxBorderY = mvImagePyramid[level].rows-_config._edge_threshold+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(_config._nfeatures*10);

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);

            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);

            for(int i=0; i<nRows; i++)
            {
                const float iniY =minBorderY+i*hCell;
                float maxY = iniY+hCell+6;

                if(iniY>=maxBorderY-3)
                    continue;
                if(maxY>maxBorderY)
                    maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    const float iniX =minBorderX+j*wCell;
                    float maxX = iniX+wCell+6;
                    if(iniX>=maxBorderX-6)
                        continue;
                    if(maxX>maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;
                    cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,_config._iniThFAST,true);

                    if(vKeysCell.empty())
                    {
                        cv::FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,_config._minThFAST,true);
                    }

                    if(!vKeysCell.empty())
                    {
                        for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                        {
                            (*vit).pt.x+=j*wCell;
                            (*vit).pt.y+=i*hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }

                }
            }

            vector<cv::KeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(_config._nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

            const int scaledPatchSize = _config._patch_size*mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for(int i=0; i<nkps ; i++)
            {
                keypoints[i].pt.x+=minBorderX;
                keypoints[i].pt.y+=minBorderY;
                keypoints[i].octave=level;
                keypoints[i].size = scaledPatchSize;
            }
        }

        // compute orientations
        for (int level = 0; level < _config._nlevels; ++level)
            computeOrientation(mvImagePyramid[level],_config, allKeypoints[level], umax);
    }

    void Detector_orb::ComputePyramid(cv::Mat image) {
        for (int level = 0; level < _config._nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            cv::Size wholeSize(sz.width + _config._edge_threshold*2, sz.height + _config._edge_threshold*2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(cv::Rect(_config._edge_threshold, _config._edge_threshold, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, _config._edge_threshold, _config._edge_threshold,
                               _config._edge_threshold, _config._edge_threshold,
                               cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, _config._edge_threshold, _config._edge_threshold,
                               _config._edge_threshold, _config._edge_threshold,
                               cv::BORDER_REFLECT_101);
            }
        }
    }
}