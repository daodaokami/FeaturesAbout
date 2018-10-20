//
// Created by lut on 18-10-12.
//

#include <chrono>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "../include/detector_sift.h"
#include <Eigen/Core>
#include <gauss_blur.h>

namespace suo15features{

    Detector_sift::Detector_sift(const suo15features::Detector_sift::Options &options)
    :sift_options(options){
        if(this->sift_options.min_octave < -1 ||
                this->sift_options.min_octave > this->sift_options.max_octave)
            throw std::invalid_argument("Invalid octave range");
        if(this->sift_options.contrast_threshold < 0.0f)
            this->sift_options.contrast_threshold = 0.02f
            / static_cast<float>(this->sift_options.num_samples_per_octave);
    }

    vector<cv::KeyPoint> Detector_sift::ExtractorKeyPoints(const cv::Mat &ori_img) {
        set_image(ori_img);
        process();
        //这是算完了要清理octaves
        //要判断要不要清理这个process
        octaves.clear();
        //目前这里的keypoints is zero？where error
        vector<cv::KeyPoint> new_keypoints;
        cout<<"kps.size "<<keypoints.size();
        new_keypoints.resize(keypoints.size());
        for(int i=0; i<keypoints.size(); ++i){
            new_keypoints[i].pt.x = keypoints[i].pt.x;
            new_keypoints[i].pt.y = keypoints[i].pt.y;
        }
        return new_keypoints;
    }

    void Detector_sift::process() {
        //记录时间， 用什么函数
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        this->create_octaves();
        this->extrema_detection();
        cout<<"SIFT detector "<<this->keypoints.size()<<endl;
        this->keypoint_localization();//这里reject的有点少！！！
        cout<<"After localization and filter "<<this->keypoints.size()<<endl;
        for(size_t i=0; i<this->octaves.size(); ++i){
            this->octaves[i].dog.clear();
        }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout<<"time out "<< elapsed_seconds.count()<<"s."<<endl;
    }

    void Detector_sift::set_image(const cv::Mat &img) {
        if(img.channels() != 1 && img.channels() != 3)
            throw std::invalid_argument("gray or color image expected");
        //unsigned char 转型成为float，方便进行计算
        if(img.channels() == 3) {
            this->orig.create(img.rows, img.cols, CV_32FC3);//float类型
            img.convertTo(orig, CV_32FC3);
        }
        else if(img.channels() == 1) {
            this->orig.create(img.rows, img.cols, CV_32FC1);
            img.convertTo(orig, CV_32FC1);
        }
        else {
            cerr << "error not suitable channels!\n";
            return;
        }
        //输入图像类型转换成float好进行接下来的计算
        //只处理灰度图，通道为1

        if(this->orig.channels() == 3){
            vector<cv::Mat> splitChannelsMat(this->orig.channels());
            cv::split(this->orig, splitChannelsMat);
            this->orig.create(img.rows, img.cols, CV_32FC1);
            this->orig = (splitChannelsMat[0]+splitChannelsMat[1]+splitChannelsMat[2])/3;
        }
        this->orig = this->orig/255.0f;
        //cout<<this->orig.row(0)<<endl;
    }

    //创建gauss prymaid！！！
    void Detector_sift::create_octaves() {
        this->octaves.clear();
        /*
         * 当初始值<0.则图像数据根据插值增大，扩大一倍的数据
         * */
        if(this->sift_options.min_octave < 0){
            float scale = 0.5;//放大尺寸的，以小于0的初值来说
            cv::Size sz(cvRound((float)orig.cols/scale), cvRound((float)orig.rows/scale));
            cv::Mat img;
            cv::resize(orig, img, sz);
            //尺寸放大一倍，不进行高斯模糊的原图/*
            //
            //  the much important function!!! */
            this->add_octave(img, this->sift_options.inherent_blur_sigma*2.0f,
                this->sift_options.base_blur_sigma);

        }
        cv::Mat img = this->orig;
        for(int i=0; i<this->sift_options.min_octave; ++i){
            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            cv::resize(img, img, sz);//并没有运行！！！
        }//降采样一半, 通过这里的操作img变成了，min_octave 一般是0，要做上采样则是-1
        //Gauss_Blur GB;
        float img_sigma = this->sift_options.inherent_blur_sigma;
        for(int i = max(0, this->sift_options.min_octave);
            i <= this->sift_options.max_octave; ++i){
            //在0层的时候一个是img_sigma 一个是 base_blur_sigma
            //the other is all base_blur_sigma and base_blur_sigma
            this->add_octave(img, img_sigma, this->sift_options.base_blur_sigma);

            //每一层图像降采样，大小变小一半
            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            cv::Mat pre_base = octaves[octaves.size()-1].img[0];
            cv::resize(pre_base, img, sz);
            //cout<<img.size()<<endl<<"resize before \n";
            //cout<<img.row(0)<<endl;
            //img = GB.rescale_half_size_gaussian(pre_base);
            //cv::resize(img, img, sz);
            //cout<<img.size()<<endl<<"resize after \n";
            //cout<<img.row(0)<<endl;
            img_sigma = this->sift_options.base_blur_sigma;
        }

    }

    void Detector_sift::add_octave(cv::Mat &image, float has_sigma, float target_sigma) {

        //cout<<"before blur \n"<<image.row(0)<<endl;
        float sigma = std::sqrt(pow(target_sigma, 2) - pow(has_sigma, 2));
        /*std::cout << "Pre-blurring image to sigma " << target_sigma << " (has "
            << has_sigma << ", blur = " << sigma << ")..." << std::endl;*/
        int ks = ceil(sigma*2.884f);
        cv::Size sz(2*ks+1, 2*ks+1);
        /*cv::Mat base = (target_sigma > has_sigma
                                      ? cv::GaussianBlur(image, image, sz, sigma, sigma)
                                      : image.clone());*/
        cv::Mat base;
        cout<<"---------------\n";
        cout<<image.row(0)<<endl;
        cout<<image.row(1)<<endl;
        cout<<image.row(2)<<endl;
        cout<<"---------------\n";
        if(target_sigma > has_sigma){
            cv::GaussianBlur(image, base, sz, sigma, sigma, cv::BORDER_REPLICATE);
            //cout<<"after blur "<<sigma<<"\n"<<base.row(0)<<endl;
        } else
            base = image.clone();
        //对第一个来说是这样的，然后要对后面的就不一样了！！！
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
            //cout<<blur_sigma<<endl;
            int cks = ceil(blur_sigma*2.884f);
            cv::Size csz(2*cks+1, 2*cks+1);
            cv::Mat img;
            cv::GaussianBlur(base, img, csz, blur_sigma, blur_sigma, cv::BORDER_REPLICATE);
            oct.img.push_back(img);

            //cout<<img.row(0)<<endl;
            /* Create the Difference of Gaussian image (DoG). */
            //计算差分拉普拉斯 // todo revised by sway
            cv::Mat dog = (img - base);
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
                size_t detector_size = this->extrema_detection(samples, static_cast<int>(i)+
                    this->sift_options.min_octave, s);
                cout<<"size_t "<<detector_size<<endl;
            }
        }
    }

    size_t Detector_sift::extrema_detection(cv::Mat *s, int oi, int si) {
        const int w = s[1].cols;
        const int h = s[1].rows;

        /*static int count = 0;
        for(int i=0; i<3; i++){
            cv::imshow("s[i]"+to_string(count), s[i]);
            cv::waitKey(0);
        }
        count++;*/
        //x是恒坐标，y是纵坐标
        int noff_x[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        int noff_y[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
        //一个点的九个方向，进行特征点的提取
        int detected = 0;
        int off = w;
        for (int y = 1; y < h - 1; ++y, off += w)
            for (int x = 1; x < w - 1; ++x)
            {//遍历像素26个像素比较，看是否是极值
                bool largest = true;
                bool smallest = true;
                float center_value = s[1].at<float>(y, x);
                for(int l=0; (largest||smallest)&&l<3; ++l){
                    for(int i=0; (largest||smallest)&&i<9; ++i){
                        if(l == 1 && i == 4)
                            continue;
                        if(s[l].at<float>(y+noff_y[i], x+noff_x[i]) >= center_value)
                            largest = false;
                        if(s[l].at<float>(y+noff_y[i], x+noff_x[i]) <= center_value)
                            smallest = false;
                    }
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

            for(int j=0; j<5; ++j){
                Dx = (dogs[1].at<float>(iy, ix+1) - dogs[1].at<float>(iy, ix-1)) * 0.5f;
                Dy = (dogs[1].at<float>(iy+1, ix) - dogs[1].at<float>(iy-1, ix)) * 0.5f;
                Ds = (dogs[2].at<float>(iy, ix) - dogs[0].at<float>(iy, ix)) * 0.5f;

                Dxx = (dogs[1].at<float>(iy, ix+1) + dogs[1].at<float>(iy, ix-1)
                        - 2.0f * dogs[1].at<float>(iy, ix));
                Dyy = (dogs[1].at<float>(iy+1, ix) + dogs[1].at<float>(iy -1, ix)
                        - 2.0f * dogs[1].at<float>(iy, ix));
                Dss = (dogs[2].at<float>(iy, ix) + dogs[0].at<float>(iy, ix)
                        - 2.0f * dogs[1].at<float>(iy, ix));

                Dxy = (dogs[1].at<float>(iy+1, ix+1) + dogs[1].at<float>(iy-1, ix-1)
                        - dogs[1].at<float>(iy+1, ix-1) - dogs[1].at<float>(iy-1, ix+1)) * 0.25f;
                Dxs = (dogs[2].at<float>(iy, ix+1) + dogs[0].at<float>(iy, ix-1)
                        - dogs[2].at<float>(iy, ix-1) - dogs[0].at<float>(iy, ix+1)) * 0.25f;
                Dys = (dogs[2].at<float>(iy+1, ix) + dogs[0].at<float>(iy-1, ix)
                        - dogs[2].at<float>(iy-1, ix) - dogs[0].at<float>(iy+1, ix)) * 0.25f;
                Eigen::Matrix3f H;
                H << Dxx, Dxy, Dxs,
                     Dxy, Dyy, Dys,
                     Dxs, Dys, Dss;
                Eigen::Vector3f b;
                b << -Dx, -Dy, -Ds;
                float detH = H.determinant();
                if(MATH_EPSILON_EQ(detH, 0.0f, 1e-15f)){
                    num_singular += 1;
                    delta_x = delta_y = delta_s = 0.0f;
                    break;
                }
                Eigen::Vector3f delta = H.ldlt().solve(b);
                delta_x = delta[0];
                delta_y = delta[1];
                delta_s = delta[2];

                int dx = (delta_x > 0.6f && ix < w-2)*1 + (delta_x<-0.6f && ix > 1)*-1;
                int dy = (delta_y > 0.6f && iy < h-2)*1 + (delta_y<-0.6f && iy > 1)*-1;

                if(dx != 0 || dy != 0){
                    ix += dx;
                    iy += dy;
                    continue;
                }
                break;
            }
            //问题处在dog上面，因为得到的Dog值，与标准的差别较大
            //cout<<dogs[1].at<float>(iy, ix)<<endl;
            float val = dogs[1].at<float>(iy, ix) + 0.5f * (Dx * delta_x + Dy * delta_y + Ds * delta_s);
            float hessian_trace = Dxx + Dyy;
            float hessian_det = Dxx * Dyy - pow(Dxy, 2);
            float hessian_score = pow(hessian_trace, 2) / hessian_det;
            float score_thres = pow(this->sift_options.edge_ratio_threshold + 1.0f, 2)
                                / this->sift_options.edge_ratio_threshold;

            kp.pt.x = (float)ix + delta_x;
            kp.pt.y = (float)iy + delta_y;
            kp.sample = (float)is + delta_s;
            //cout<<(val)<<", "<<sift_options.contrast_threshold<<endl;
            /*cout<<abs(val)<<", "<<hessian_score<<", "<<
                abs(delta_x)<<", "<<abs(delta_y)<<", "<<abs(delta_s)<<endl;*/
            //cout<<abs(val)<<" "<<this->sift_options.contrast_threshold<<endl;
            if (std::abs(val) < this->sift_options.contrast_threshold
                || hessian_score < 0.0f || hessian_score > score_thres
                || std::abs(delta_x) > 1.5f || std::abs(delta_y) > 1.5f || std::abs(delta_s) > 1.0f
                || kp.sample < -1.0f
                || kp.sample > (float)this->sift_options.num_samples_per_octave
                || kp.pt.x < 0.0f || kp.pt.x > (float)(w - 1)
                || kp.pt.y < 0.0f || kp.pt.y > (float)(h - 1)){
                //cout<<"reject"<<endl;
                continue;
            }
            this->keypoints[num_keypoints] = kp;
            num_keypoints += 1;
        }

        this->keypoints.resize(num_keypoints);

    }

}
