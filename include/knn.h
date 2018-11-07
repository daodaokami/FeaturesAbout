//
// Created by lut on 18-10-30.
//

#ifndef FASTKNN_KNN_H
#define FASTKNN_KNN_H

#include "distance.h"
namespace math_tools {
    template <typename T>
    struct Result{
        T dist_1st_best;
        T dist_2nd_best;
        int index_1st_best;
        int index_2nd_best;

        Result(){}
        Result(T d1_best, T d2_best, int i1_best, int i2_best):dist_1st_best(d1_best), dist_2nd_best(d2_best),
                                                               index_1st_best(i1_best), index_2nd_best(i2_best){}

    };

    template <typename T>
    class Knn {//只保存计算相关的参数
    private:
        Distance_options options;
        Distance<T>* distance_measurement;
        int sample_nums;
    public:
        Knn(const Distance_options& opts, int nums):options(opts), distance_measurement(nullptr), sample_nums(nums){
            distance_measurement = new Distance<T>(opts);
        }
        /*
         * query 是当前被访问的对象，samples是查找的对象，要对这个对象进行遍历，寻找最优和次优的距离
         * 比例小于一定的阈值
         * */
        void search_knnsample(T const* query, T const* samples, Result<T>* result);//返回的是

        ~Knn(){
            //cout<<"delete distance_measurement"<<endl;
            delete distance_measurement;
        }
    };
    template<typename T>
    void Knn<T>::search_knnsample(T const*query, T const*samples, Result<T>* result){
        /*
         * distance init with bigest size
         * */
        if(result == nullptr)
        {
            std::cerr<<"do not have space to save the result"<<std::endl;
        }
        //小的值表示对应的相似度越好，这里与cos相似度是不同的（越接近1则相似度越大）
        result->dist_2nd_best = static_cast<T>(256);
        result->dist_1st_best = static_cast<T>(256);
        vector<int> diss;
        diss.resize(sample_nums);
        //在o（n）的时间复杂度下面得到一个最优解和一个次优解
        for(int i=0; i<this->sample_nums; i++){
            T inner_product = static_cast<T>(0);
            int offset = i*this->options.dimension/32;//因为最大是256维度的数据，int 4个字节，uchar 1个字节
            T const* offsample = &(samples[offset]);
            inner_product = distance_measurement->cal_hamming_distance(query, offsample);
            diss[i] = inner_product;
            //std::cout<<inner_product<<std::endl;
            if(inner_product <= result->dist_2nd_best){
                if(inner_product <= result->dist_1st_best){
                    result->index_2nd_best = result->index_1st_best;
                    result->dist_2nd_best = result->dist_1st_best;
                    result->index_1st_best = i;
                    result->dist_1st_best = inner_product;
                }
                else{
                    result->index_2nd_best = i;
                    result->dist_2nd_best = inner_product;
                }
            }
        }
        /*static bool flag = true;
        if(flag) {
            for (int i = 0; i < sample_nums; i++) {
                cout << i << " " << diss[i] << endl;
            }
            flag = false;
        }*/
    }
}


#endif //FASTKNN_KNN_H
