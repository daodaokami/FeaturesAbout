//
// Created by lut on 18-10-12.
//

#include <chrono>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
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
        //unsigned char 转型成为float，方便进行计算
        this->orig.create(img.rows, img.cols, CV_32FC1);//float类型
        //输入图像类型转换成float好进行接下来的计算
        //只处理灰度图，通道为1
        img.convertTo(orig, CV_32FC1);
    }

    //创建gauss prymaid！！！
    void Detector_sift::create_octaves() {
        this->octaves.clear();
        /*
         * 当初始值<0.则图像数据根据插值增大，扩大一倍的数据
         * */
        if(this->sift_options.min_octave < 0){
            float scale = 0.5;//放大尺寸的，以小于0的初值来说
            cv::Size sz(cvRound((float)orig.cols*scale), cvRound((float)orig.rows*scale));
            cv::Mat img;
            cv::resize(orig, img, sz);
            //尺寸放大一倍，不进行高斯模糊的原图
            this->add_octave(img, this->sift_options.inherent_blur_sigma*2.0f,
                this->sift_options.base_blur_sigma);
        }
        cv::Mat img = this->orig;
        for(int i=0; i<this->sift_options.min_octave; ++i){
            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            cv::resize(img, img, sz);
        }//降采样一半

        float img_sigma = this->sift_options.inherent_blur_sigma;
        for(int i = max(0, this->sift_options.min_octave);
            i <= this->sift_options.max_octave; ++i){
            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            this->add_octave(img, img_sigma, this->sift_options.base_blur_sigma);

            cv::Mat pre_base = octaves[octaves.size()-1].img[0];
            cv::resize(img, img, sz);
            img_sigma = this->sift_options.base_blur_sigma;
        }

    }

    void Detector_sift::add_octave(cv::Mat &image, float has_sigma, float target_sigma) {
        float sigma = std::sqrt(pow(target_sigma, 2) - pow(has_sigma, 2));
        //std::cout << "Pre-blurring image to sigma " << target_sigma << " (has "
        //    << has_sigma << ", blur = " << sigma << ")..." << std::endl;
        cv::Size sz(sigma*2.884f+1, sigma*2.884f+1);
        /*cv::Mat base = (target_sigma > has_sigma
                                      ? cv::GaussianBlur(image, image, sz, sigma, sigma)
                                      : image.clone());*/
        cv::Mat base;
        if(target_sigma > has_sigma){
            cv::GaussianBlur(image, base, sz, sigma, sigma);
        } else
            base = image.clone();
        /* Create the new octave and add initial image. */
        this->octaves.push_back(Octave());
        Octave& oct = this->octaves.back();
        oct.img.push_back(base);

        /* 'k' is the constant factor between the scales in scale space. */
        float const k = std::pow(2.0f, 1.0f / this->sift_options.num_samples_per_octave);
        sigma = target_sigma;

        /* Create other (s+2) samples of the octave to get a total of (s+3). */
        for (int i = 1; i < this->sift_options.num_samples_per_octave + 3; ++i)
        {
            /* Calculate the blur sigma the image will get. */
            float sigmak = sigma * k;
            float blur_sigma = std::sqrt(pow(sigmak, 2) - pow(sigma, 2));

            /* Blur the image to create a new scale space sample. */
            //std::cout << "Blurring image to sigma " << sigmak << " (has " << sigma
            //    << ", blur = " << blur_sigma << ")..." << std::endl;
            cv::Size sz(blur_sigma*2.884f, blur_sigma*2.884f);
            cv::Mat img;
            cv::GaussianBlur(base, img, sz, blur_sigma, blur_sigma);

            oct.img.push_back(img);

            /* Create the Difference of Gaussian image (DoG). */
            //计算差分拉普拉斯 // todo revised by sway
            cv::Mat dog = (img - dog);
            oct.dog.push_back(dog);

            /* Update previous image and sigma for next round. */
            base = img;
            sigma = sigmak;
        }
    }

    void Detector_sift::extrema_detection() {
        this->keypoints.clear();

        for(size_t i=0; i<this->octaves.size(); ++i){
            const Octave& oct(this->octaves[i]);
            for(int s=0; s<(int)oct.dog.size()-2; ++s){
                cv::Mat samples[3] =
                        {oct.dog[s+0], oct.dog[s+1], oct.dog[s+2]};
                this->extrema_detection(samples, static_cast<int>(i)+
                    this->sift_options.min_octave, s);
            }
        }
    }

    size_t Detector_sift::extrema_detection(cv::Mat *s, int oi, int si) {
        const int w = s[1].cols;
        const int h = s[1].rows;

        int noff[9] = {-1-w, 0-w, 1-w, -1, 0, 1, -1+w, 0+w, 1+w};


        int detected = 0;
        int off = w;
        for (int y = 1; y < h - 1; ++y, off += w)
            for (int x = 1; x < w - 1; ++x)
            {
                int idx = off + x;

                bool largest = true;
                bool smallest = true;
                float center_value = s[1].at<float>(idx);
                for (int l = 0; (largest || smallest) && l < 3; ++l)
                    for (int i = 0; (largest || smallest) && i < 9; ++i)
                    {
                        if (l == 1 && i == 4) // Skip center pixel
                            continue;
                        // Maybe have problem!!!
                        if (s[l].at<float>(idx + noff[i]) >= center_value)
                            largest = false;
                        if (s[l].at<float>(idx + noff[i]) <= center_value)
                            smallest = false;
                    }

                /* Skip non-maximum values. */
                if (!smallest && !largest)
                    continue;

                /* Yummy. Add detected scale space extremum. */
                Sift_KeyPoint kp;
                kp.octave = oi;
                kp.pt.x = static_cast<float>(x);
                kp.pt.y = static_cast<float>(y);
                kp.sample = static_cast<float>(si);

                this->keypoints.push_back(kp);
                detected += 1;
            }

        return detected;
    }

    void Detector_sift::keypoint_localization()
    {
        int num_singular = 0;
        int num_keypoints = 0; // Write iterator
        for (std::size_t i = 0; i < this->keypoints.size(); ++i) {
            /* Copy keypoint. */
            Sift_KeyPoint kp(this->keypoints[i]);

            /* Get corresponding octave and DoG images. */
            Octave const &oct(this->octaves[kp.octave - this->sift_options.min_octave]);
            int sample = static_cast<int>(kp.sample);
            cv::Mat dogs[3] = {oct.dog[sample + 0], oct.dog[sample + 1], oct.dog[sample + 2]};

            /* Shorthand for image width and height. */
            int const w = dogs[0].cols;
            int const h = dogs[0].rows;
            /* The integer and floating point location of the keypoints. */
            int ix = static_cast<int>(kp.pt.x);
            int iy = static_cast<int>(kp.pt.y);
            int is = static_cast<int>(kp.sample);
            float delta_x, delta_y, delta_s;
            /* The first and second order derivatives. */
            float Dx, Dy, Ds;
            float Dxx, Dyy, Dss;
            float Dxy, Dxs, Dys;


        }

        this->keypoints.resize(num_keypoints);
    }
    void Detector_sift::descriptor_generation() {}

    void Detector_sift::generate_grad_ori_images(suo15features::Detector_sift::Octave *octave) {}

    void Detector_sift::orientation_assignment(Sift_KeyPoint const &kp,
                                               const suo15features::Detector_sift::Octave *octave,
                                               vector<float> &orientations) {

    }

    bool Detector_sift::descriptor_assignment(Sift_KeyPoint const &kp, cv::Mat &desc,
                                              const suo15features::Detector_sift::Octave *octave) {


    }

    float Detector_sift::keypoint_relative_scale(const Sift_KeyPoint &kp) {
        return this->sift_options.base_blur_sigma*std::pow(2.0f,
            (kp.sample+1.0f)/this->sift_options.num_samples_per_octave);
    }

    float Detector_sift::keypoint_absolute_scale(const Sift_KeyPoint &kp) {
        return this->sift_options.base_blur_sigma*pow(2.0f,
            kp.octave+(kp.sample+1.0f)/this->sift_options.num_samples_per_octave);
    }



}
