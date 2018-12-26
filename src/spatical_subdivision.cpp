//
// Created by lut on 18-12-20.
//

#include "spatical_subdivision.h"
#include <iostream>

struct Rect_Area{
    cv::Point2i start_pt;
    int width;
    int height;
    Rect_Area(){
        start_pt = cv::Point2i(0,0);
        width = height = 0;
    }

    Rect_Area(cv::Point2i pt, int w, int h):
    start_pt(pt), width(w), height(h){}
};
Spatical_Subdivision::Spatical_Subdivision() {
    this->border_w = 0;
    this->border_h = 0;
    this->vIndexes.clear();

}

Spatical_Subdivision::Spatical_Subdivision(cv::Size img_sz, int wh) {
    //看图像的大小能够分成多大的尺寸
    //确定好图像的像素划分,最好进行均匀的像素划分
    this->wh = wh;
    int h = img_sz.height;
    int resi_h = h%wh;
    this->border_h = resi_h/2;//如果resi_h是奇数的话,直接向下取整就好
    //把top和bottom的数据存在以resi_h/2为高划分的小区域内,一般情况下可以认为是边缘点,经常会被省略
    int count_h = h/wh;
    if(resi_h != 0)
        count_h += 2;
    this->r = count_h;
    int w = img_sz.width;
    int resi_w = w%wh;
    int count_w = w/wh;
    if(resi_w != 0)
        count_w += 2;
    this->c = count_w;
    this->border_w = resi_w/2;
    int sum_subgrid = count_h*count_w;
    //把left和right的数据存放在resi_w/2的小范围的区间内,这样,边界的长宽可定,范围是0-wh/2的范围,右开区间
    if(!vIndexes.empty()){
        vIndexes.clear();
    }
    this->vIndexes.resize(sum_subgrid);
}

void Spatical_Subdivision::SplitPoints2Index(cv::Size img_sz, std::vector<cv::KeyPoint>& ps) {
    std::cout<<"space dividing!"<<std::endl;
    int sum_grid = static_cast<int>(this->vIndexes.size());
    int avg_point_ingrid = static_cast<int>(ceil(ps.size()/sum_grid));
    for(int i=0; i<sum_grid; i++){
        this->vIndexes[i].reserve(avg_point_ingrid);
    }
    int count_h=img_sz.height/this->wh, count_w=img_sz.width/this->wh;
    std::vector<Rect_Area> rect_areas;
    rect_areas.resize(sum_grid);
    bool flag_h = false, flag_w = false;
    if(border_h != 0) {
        count_h += 2;
        flag_h = true;
    }
    if(border_w != 0) {
        count_w += 2;
        flag_w = true;
    }
    int dx, dy;
    if(flag_h)
        dy = border_h;
    else
        dy = wh;
    if(flag_w)
        dx = border_w;
    else
        dx = wh;
    /**
     * 确认划分空间的位置
     * 即点的位置信息
     * */
    for(int i=0; i<sum_grid; i++) {
        //判断是否是边界
        int row = i / count_w, col = i % count_w;
        cv::Point2i start_pt;
        int wi = wh, hi = wh;
        if ((row == 0 || row == count_h - 1) && (col == 0 || col == count_w - 1)) {
            //四个角上时
            wi = border_w, hi = border_h;
            if (row == 0) {
                if (col == 0) {
                    start_pt.x = 0;
                    start_pt.y = 0;
                } else {
                    start_pt.x = dx + wh * (count_w - 2);
                    start_pt.y = 0;
                }
            } else {//row 在bottom上
                if (col == 0) {
                    start_pt.x = 0;
                    start_pt.y = dy + wh * (count_h - 2);
                } else {
                    start_pt.x = dx + wh * (count_w - 2);
                    start_pt.y = dy + wh * (count_h - 2);
                }
            }
        } else if ((row == 0 || row == count_h - 1) && (col != 0 || col != count_w - 1)) {
            //说明在第一行或者最后一行
            hi = border_h;
            if (row == 0) {
                start_pt.x = dx + wh * (col - 1);
                start_pt.y = 0;
            } else {
                start_pt.x = dx + wh * (col - 1);
                start_pt.y = dy + wh * (count_h - 2);
            }

        } else if ((row != 0 && row != count_h - 1) && (col == 0 || col == count_w - 1)) {
            //在第一列或者最后一列
            wi = border_w;
            if (col == 0) {
                start_pt.x = 0;
                start_pt.y = dy + wh * (row - 1);
            } else {
                start_pt.x = dx + wh * (count_w - 2);
                start_pt.y = dy + wh * (row - 1);
            }
        } else {
            //中间模块,不用修改边界大小的问题
            start_pt.x = dx + (col - 1) * wh;
            start_pt.y = dy + (row - 1) * wh;
        }
        rect_areas[i].start_pt = start_pt;
        rect_areas[i].height = hi;
        rect_areas[i].width = wi;
    }

    /*
     * 通过遍历特征点,为其分配对应的索引
     *
     * */
    for(int i=0; i<ps.size(); i++){
        //将点划分到每个网格中去
        int col = 0, row = 0;
        if(ps[i].pt.x >= border_w){
            if(border_w != 0)
                col = 1;
            col += (ps[i].pt.x - border_w)/wh;
            //例 110, x=110, col-20 / 80 = 1, 说明在横向的第三个检索块中.
            //用编号从0 开始,应该是1
        }
        if(ps[i].pt.y >= border_h){
            if(border_h != 0)
                row = 1;
            row += (ps[i].pt.y - border_h)/wh;
            //也是从编号0,开始
        }
        //有了所在行列,那么对应的直线数据编号为
        int index = row*count_w + col;
        this->vIndexes[index].push_back(i);

        //std::cout<<ps[i].pt.x <<" "<<ps[i].pt.y<<" index "<<index<<std::endl;
        //然后查看,划分好空间的点的位置,并通过在图像上显示来查看是否正确
        //存储的是特征点的索引序号
        /*
         *  在外部搜索,并提供,相应周围的8个邻域的特征点的索引号
         *
         * */
    }
}

int Spatical_Subdivision::GetGridNofromKeypoints(cv::KeyPoint& kp){
    int col = 0, row = 0;
    if(kp.pt.x >= border_w){
        col += (kp.pt.x - border_w)/wh;
        //例 110, x=110, col-20 / 80 = 1, 说明在横向的第二个检索块中.
        //用编号从0 开始,应该是1
    }
    if(kp.pt.y >= border_h){
        row += (kp.pt.y - border_h)/wh;
    }
    //有了所在行列,那么对应的直线数据编号为,在都未知的情况下,该怎么做
    int index = row*this->c + col;
    return index;
}//就能够返回有效的局特征点

//缩小匹配范围的尺度,编程的时候,可以直接用块内的所有点与对应的描述子,和其周围空间的进行比较,用opencv自带的代码,查看加速的效率
//以及通过这样的加速后,使用GMS是否任然有用?这是实验的重点!!!
std::vector<int> Spatical_Subdivision::GetNeighbors(int cur_index) {
    //如何从vector中取指定位置的值,返回一个新的vector呢
    //return the cur_pos and the neighbors
    int row = cur_index/this->c;
    int col = cur_index%this->c;
    //周围9个点,当col和row是边界的时候,就会少很多内容,如果是单位边界
    int cur_neighbor[9] = {-this->c-1, -this->c, -this->c+1, -1, 0, 1, this->c-1, this->c, this->c+1};
    std::vector<int> vcur_neighbor;
    vcur_neighbor.reserve(9);
    //九个位置,当处于某些边界的时候需要去掉,不能计算进去
    //四个角点,去除5个点位置
    if(row == 0){
        if(col == 0){
            vcur_neighbor.resize(4);
            vcur_neighbor[0] = cur_neighbor[4];
            vcur_neighbor[1] = cur_neighbor[5];
            vcur_neighbor[2] = cur_neighbor[7];
            vcur_neighbor[3] = cur_neighbor[8];
        }else if(col == this->c-1){
            vcur_neighbor.resize(4);
            vcur_neighbor[0] = cur_neighbor[3];
            vcur_neighbor[1] = cur_neighbor[4];
            vcur_neighbor[2] = cur_neighbor[6];
            vcur_neighbor[3] = cur_neighbor[7];
        }else{
            vcur_neighbor.resize(6);
            for(int i=0; i<6; i++){
                vcur_neighbor[i] = cur_neighbor[3+i];
            }
        }
    }else if(row == this->r-1){
        if(col == 0){
            vcur_neighbor.resize(4);
            vcur_neighbor[0] = cur_neighbor[1];
            vcur_neighbor[1] = cur_neighbor[2];
            vcur_neighbor[2] = cur_neighbor[4];
            vcur_neighbor[3] = cur_neighbor[5];
        }else if(col == this->c-1){
            vcur_neighbor.resize(4);
            vcur_neighbor[0] = cur_neighbor[0];
            vcur_neighbor[1] = cur_neighbor[1];
            vcur_neighbor[2] = cur_neighbor[3];
            vcur_neighbor[3] = cur_neighbor[4];
        }else{
            vcur_neighbor.resize(6);
            for(int i=0; i<6; i++){
                vcur_neighbor[i] = cur_neighbor[i];
            }
        }
    }else{//中间行
        if(col == 0){
            vcur_neighbor.resize(6);
            vcur_neighbor[0] = cur_neighbor[1];
            vcur_neighbor[1] = cur_neighbor[2];
            vcur_neighbor[2] = cur_neighbor[4];
            vcur_neighbor[3] = cur_neighbor[5];
            vcur_neighbor[4] = cur_neighbor[7];
            vcur_neighbor[5] = cur_neighbor[8];
        }else if(col == this->c-1){
            vcur_neighbor.resize(6);
            vcur_neighbor[0] = cur_neighbor[0];
            vcur_neighbor[1] = cur_neighbor[1];
            vcur_neighbor[2] = cur_neighbor[3];
            vcur_neighbor[3] = cur_neighbor[4];
            vcur_neighbor[4] = cur_neighbor[6];
            vcur_neighbor[5] = cur_neighbor[7];
        }else{
            vcur_neighbor.resize(9);
            for(int i=0; i<9; i++){
                vcur_neighbor[i] = cur_neighbor[i];
            }
        }
    }
    //计算完了邻域块的位置
    return vcur_neighbor;
}

void Spatical_Subdivision::ComputeGirdAndNeighbors(cv::Size img_sz, std::vector<cv::KeyPoint> &ps,
                                                   cv::Mat &descs,
                                                   GridNet &gridNet) {

    /*
     * 1.对图像空间进行网格划分
     *
     * 2.将特征点,映射到网格空间中,Mat也要重新创建
     *
     * 3.为了编程方便,直接查看特征匹配的速度,先得到的是特征点映射的网格空间中的索引,在批量的处理keypoints和descs的划分
     *
     * */
    if(!this->vIndexes.empty())
        this->vIndexes.clear();
    this->vIndexes.resize(this->c*this->r);
    this->SplitPoints2Index(img_sz, ps);
    //this->vIndexes;//通过vIndexes来得到特征点的集合
    int gridSize = static_cast<int>(this->vIndexes.size());
    gridNet.gridKeypoints.resize(gridSize);
    gridNet.gridDescs.resize(gridSize);
    gridNet.neighbors.resize(gridSize);
    gridNet.descs_neighbors.resize(gridSize);
    for(int i=0; i < gridSize; i++){
        int pointSize = static_cast<int>(vIndexes[i].size());
        gridNet.gridKeypoints[i].resize(pointSize);//初始化每个网格空间的大小
        //gridNet.gridDescs size is
        //这时候,没有初始化的网格空间,是否会有固定的地址? 不会有的在初始化完成之后,进行
        // 网格的net链接
        for(int j=0; j<gridNet.gridKeypoints[i].size(); j++){
            gridNet.gridKeypoints[i][j]= &ps[vIndexes[i][j]];
            gridNet.gridDescs[i].push_back(descs.row(vIndexes[i][j]));//描述子按照行来存储!Mat 行直接赋值是浅/深拷贝
            //这里已经存了所有块对应的描述子了
        }
    }

    for(int i=0; i<gridSize; i++){
        //判断有多少邻居
        std::vector<int> vcur_neighbors = GetNeighbors(i);
        int neighbor_size = static_cast<int>(vcur_neighbors.size());
        //data struct
        //vector<vector<vector<keyppoints>*>>
        gridNet.neighbors[i].resize(neighbor_size);
        gridNet.descs_neighbors[i].resize(neighbor_size);
        for(int j=0; j<neighbor_size; j++){
            gridNet.neighbors[i][j] = (&gridNet.gridKeypoints[i+vcur_neighbors[j]]);
            gridNet.descs_neighbors[i][j] = (gridNet.gridDescs[i+vcur_neighbors[j]]);
        }
    }
    std::cout<<"data have been grided!"<<std::endl;
    //好了网格构造完成,开始测试,网格的正确与否!!
}

void Spatical_Subdivision::ComputeGridAndNeighbors(cv::Size img_sz, std::vector<cv::KeyPoint> &ps, cv::Mat &descs,
                                                   GridNet_No &gridNet_no) {
    //只存储新的


}
Spatical_Subdivision::~Spatical_Subdivision() {
    //要对网格的数据,进行析构
    std::cout<<"delete the 指针的数据?"<<std::endl;
    /*
     * 注意这里对于kps_neighbors 不用析构,因为这只是传了指针
     * 而对于cv::Mat 的数组,不清楚这里的条件是深拷贝还是浅拷贝
     * */

}