//
// Created by lut on 18-10-29.
//

#ifndef FASTKNN_DISTANCE_H
#define FASTKNN_DISTANCE_H
#include<iostream>
#include <cmath>
#define MATH_POW2(x) ((x) * (x))

namespace math_tools {
    enum{
        EUCLIDEAN_DISTANCE = 0,
        MANHATTAN_DISTANCE = 1,
        HAMMING_DISTANCE = 2,
        COSINE_SIMILARITY = 3
    };

    struct Distance_options{
        int dimension;
        int dis_type;

        Distance_options(void){}
        Distance_options(const Distance_options& opts){
            dimension = opts.dimension;
            dis_type = opts.dis_type;
        }
        Distance_options(int dime, int type):dimension(dime), dis_type(type){}
    };

    template<typename T>
    class Distance {//支持float和int类型的计算！
    private:
        Distance_options options;

    public:
        void show(){
            std::cout<<"this func is linked !"<<std::endl;
        }
        Distance(){}
        Distance(const Distance_options& opts):options(opts){}

        T calculate_distance(T const* queryIdx, T const* trainIdx);

        T cal_euclidean_distance(T const* queryIdx, T const* trainIdx);
        T cal_manhattan_distance(T const* queryIdx, T const* trainIdx);
        T cal_hamming_distance(T const* queryIdx, T const* trainIdx);
        /* at cos the T* should be normlized(1)
         * */
        T cal_cosine_similariry(T const* queryIdx, T const* trainIdx);
    };
    template<typename T>
    T Distance<T>::calculate_distance(T const*queryIdx, T const*trainIdx) {
        T distance = static_cast<float>(0);
        //U 可选的类型为float 和 int（专门为hanmming距离准备的的int类型）
        switch(this->options.dis_type){
            case EUCLIDEAN_DISTANCE:
                distance = (cal_euclidean_distance(queryIdx, trainIdx));
                break;
            case HAMMING_DISTANCE:
                distance = (cal_hamming_distance(queryIdx, trainIdx));
                break;
            case MANHATTAN_DISTANCE:
                distance = (cal_manhattan_distance(queryIdx, trainIdx));
                break;
            case COSINE_SIMILARITY:
                distance = (cal_cosine_similariry(queryIdx, trainIdx));
                break;
            default:
                break;
        }
        return (distance);
    }

    template<typename T>
    T Distance<T>::cal_euclidean_distance(T const* queryIdx, T const* trainIdx){
        T dist = static_cast<T>(0);
        for(int i=0; i<this->options.dimension; i++){
            dist += MATH_POW2(queryIdx[i] - trainIdx[i]);
        }
        return sqrt(dist);
    }
    template<typename T>
    T Distance<T>::cal_manhattan_distance(T const* queryIdx, T const* trainIdx){
        T dist = static_cast<T>(0);
        for(int i=0; i<this->options.dimension; i++){
            dist += std::abs(queryIdx[i] - trainIdx[i]);
        }
        return dist;
    }

    //只能是int类型
    template<typename T>
    T Distance<T>::cal_hamming_distance(T const* queryIdx, T const* trainIdx){
        T dist=static_cast<T>(0);
        for(int i=0; i<8; i++, queryIdx++, trainIdx++)
        {
            unsigned  int v = *queryIdx ^ *trainIdx;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        //计算距离
        return dist;
    }
    /* at cos the T* should be normlized(1)
     * */
    template<typename T>
    T Distance<T>::cal_cosine_similariry(T const* queryIdx, T const* trainIdx){
        T dist = static_cast<T>(0.0);
        for(int i=0; i<this->options.dimension; ++i){
            dist += queryIdx[i] * trainIdx[i];//向量的乘积即可
        }
        return dist;
    }
}

#endif //FASTKNN_DISTANCE_H
