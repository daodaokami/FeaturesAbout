//
// Created by lut on 18-10-12.
//

#include <chrono>
#include "../include/detector_sift.h"

namespace suo15features{

    Detector_sift::Detector_sift(const suo15features::Detector_sift::Options &options)
    :sift_options(options){
        if(this->sift_options.min_octave < -1 ||
                this->sift_options.min_octave > this->sift_options.max_octave)
            throw std::invalid_argument("Invalid octave range");
        if(this->sift_options.contrast_threshold < 0.0f)
            this->sift_options.contrast_threshold == 0.02f
            / static_cast<float>(this->sift_options.num_samples_per_octave);
    }

    void
    Detector_sift::process() {
        //记录时间， 用什么函数
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        /*this->create_octaves();
        this->extrema_detection();
        this->keypoint_localozation();
        for(size_t i=0; i<this->octaves.size(); ++i){
            this->octaves[i].dog.clear();
        }
        this->descriptor_generation();
        this->octaves.clear();
*/
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;
    }

    void Detector_sift::set_image(const cv::Mat &img) {
        if(img.channels() != 1 && img.channels() != 3)
            throw std::invalid_argument("gray or color image expected");
        this->orig.create(img.rows, img.cols, CV_32FC1);//float类型
        //输入图像类型转换成float好进行接下来的计算
        //只处理灰度图，通道为1
        img.convertTo(orig, CV_32FC1);
    }

    void Detector_sift::create_octaves() {
        this->octaves.clear();
        if(this->sift_options.min_octave < 0){
            float scale = 0.5;//放大尺寸的，以小于0的初值来说
            cv::Size sz(cvRound((float)orig.cols*scale), cvRound((float)orig.rows*scale));
            cv::Mat img;
            cv::resize(orig, img, sz);
            /*this->add_octave(img, this->sift_options.inherent_blur_sigma*2.0f,
                this->sift_options.base_blur_sigma);*/
        }


    }
}