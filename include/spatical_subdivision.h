//
// Created by lut on 18-12-20.
//

#ifndef SPATIALSUBDIVISION_SPATICAL_SUBDIVISION_H
#define SPATIALSUBDIVISION_SPATICAL_SUBDIVISION_H
#include <opencv2/core/core.hpp>
#include <vector>
/*
 * 该类的设计目的是为了能够对图像和特征点,能够进行划分
 * 将图像分成大小一致的正方形,并把其中的特征点与所在区块
 * 的位置关系能够快速的找到,即
 * 1.一个特征点 ----  对应的划分区块的位置
 * 2.一个区块   ----  块中的所有特征点都要找到
 * 3.一个区块   ----  找到区块相邻的区块与其中的特征点
 * */
struct GridNet{
    typedef std::vector<cv::KeyPoint*> ptr_kps;//在gridKeyPoints 中存储的是这种数据
    typedef std::vector<ptr_kps*> vec_ptr_kps;
    typedef std::vector<cv::Mat> vec_Mat;
    std::vector<ptr_kps> gridKeypoints;
    std::vector<vec_ptr_kps> neighbors;    //存着neighbors的内容指针指向这些位置
    vec_Mat gridDescs;                     //一个块里面存放着每个网格的描述子,没有邻域描述子
    //这里的每个元素都指向上面,vector中的一个?
    //循环的次数,减少的计算根据图像的大小来决定的
    std::vector<vec_Mat> descs_neighbors;//将每个块对应的相邻的块的描述子存储起来
};
//计算邻域这里存在一些问题!!!

struct GridNet_No{
    typedef std::vector<int> v_ps;
    typedef std::vector<v_ps> v_neighbor_v_ps;
    typedef std::vector<v_ps> v_index_v_ps;
    typedef std::vector<v_neighbor_v_ps*> v_index_ptr_v_neighbor_v_ps;
    v_index_v_ps index_ps;                  //每个网格的对应结构
    v_index_ptr_v_neighbor_v_ps neighbors;  //每个网格对应的v_index_v_ps
};

//在得到了这样的网格之后,计算GridNet的匹配领域块的次数是否会减少呢???
class Spatical_Subdivision{
private:
    //数据结构来存取每个块的位置与索引
    //用map来存储(key - value)??
    //小块的长宽高,是一样的.
    int wh;//px
    int border_w, border_h;
    int c, r;
    //不采用map格式,直接用index的序列好就好,这样创建的速度快一些
    //在GMS中的划分格子大小为20个像素,这里采用80个像素大小,应该比较合适.
    /*
     * vIndexes 表示每个块内,
     * 可以存储的特征点的索引
     *
     * */
    std::vector<std::vector<int>> vIndexes;
    //将像素划分进这些块中,首先要初始化块的大小,然后进行划分,时间复杂度为O(n)
    /*
     * 0 1 2 3 4
     * 5 6 7 8 9
     * 块的序号按照如上顺序划分
     * */
public:
    //如果在进行特征点提取的时候就进行了划分,那么这个O(n)的时间基本是可以省去的
    Spatical_Subdivision();
    Spatical_Subdivision(cv::Size img_sz, int wh);

    std::vector<std::vector<int>> GetvIndexes(){
        return this->vIndexes;
    }

    void SplitPoints2Index(cv::Size img_sz, std::vector<cv::KeyPoint>& ps);//这里可能是points,也可能是keypoints
    int GetGridNofromKeypoints(cv::KeyPoint& kp);
    std::vector<int> GetNeighbors(int cur_index);
    /*如果用单一方法得到邻域,增加了提取出特征点序号的操作,但是是不可少的
     *如果要加快速度,直接在重写keypoints的matcher中,对点进行筛选就好,
     * 但是采用opencv提供的matcher来做,所以必要提取对应的特征点与描述
     * 但是最后批量操作,这样可以省去比较多的时间,可以一次就构造一堆点的
     * 邻域  最后数据存储的方式呢!得到一个网格,网格中存储了所有的特征点
     * 以及对应的邻域 */
    void ComputeGirdAndNeighbors(cv::Size img_sz, std::vector<cv::KeyPoint>& ps,
                                    cv::Mat& descs, GridNet& gridNet);

    void ComputeGridAndNeighbors(cv::Size img_sz, std::vector<cv::KeyPoint>& ps,
                                    cv::Mat& descs, GridNet_No& gridNet_no);
    ~Spatical_Subdivision();
};

#endif //SPATIALSUBDIVISION_SPATICAL_SUBDIVISION_H
