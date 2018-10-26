//
// Created by lut on 18-10-12.
//

#include "../include/desc_s128.h"

namespace suo15features{

    Desc_s128::Desc_s128(suo15features::S128_Options options):
            desc_s128_options(options) {
        if(this->desc_s128_options.min_octave < -1 ||
           this->desc_s128_options.min_octave > this->desc_s128_options.max_octave)
            throw invalid_argument("Invalid octave range");

        if(this->desc_s128_options.contrast_threshold < 0.0f)
            this->desc_s128_options.contrast_threshold = 0.02f/
                                                         static_cast<float>(this->desc_s128_options.num_samples_per_octave);
    }

    cv::Mat Desc_s128::ComputeDescriptor(const cv::Mat &image, vector<Sift_KeyPoint> &keypoints) {
        //首先这里若存在点在同一层Octave上，那么
        create_octaves(image);
       /* for(int i=0; i<this->octaves.size();i++){
            for(int j=0; j<this->octaves[i].img.size(); j++){
                cv::Mat out = octaves[i].img[j];
                cv::imshow("out", out);
               // cout<<out<<endl;
                cv::waitKey(0);
            }
        }//金字塔的创建没有明显异常*/
        descriptor_generation(keypoints);//存在异常的错误
        if(descriptors.empty())
            descriptors.resize(0);
        return this->descriptors;
    }


    void Desc_s128::descriptor_generation(vector<suo15features::Sift_KeyPoint> &keypoints) {
        vector<Sift_KeyPoint> new_keypoints;
        new_keypoints.reserve(keypoints.size()*3/2);
        if(this->octaves.empty())
            throw std::runtime_error("Octave not available!");
        if(keypoints.empty())
            return ;
        //首先申请keypoints的大小的两倍大小的， description， 描述子是按照行来存储的
        this->descriptors.release();
        this->descriptors.reserve(keypoints.size()*3/2);

        int octave_index = keypoints[0].octave;
        Simple_Octave* octave = &this->octaves[octave_index - this->desc_s128_options.min_octave];

        //todo every octave imgs grad and ori
        this->generate_grad_ori_images(octave);

        for(size_t i=0; i<keypoints.size(); i++){
            const Sift_KeyPoint& kp(keypoints[i]);

            //Generate new gradient and orientation images if octave changed
            if(kp.octave > octave_index){
                if(octave){
                    octave->grad.clear();
                    octave->ori.clear();
                }//不能乱序搜索！
                octave_index = kp.octave;
                octave = &this->octaves[octave_index - this->desc_s128_options.min_octave];
                this->generate_grad_ori_images(octave);
            }
            else if(kp.octave < octave_index){
                throw runtime_error("Decreasing octave index");
            }

            vector<float> orientations;
            orientations.reserve(8);
            this->orientation_assignment(kp, octave, orientations);

            //todo create descriptors, the same features has many descriptors
            for(size_t j=0; j<orientations.size(); j++){
                Descriptor desc;
                float const scale_factor = pow(2.0f, kp.octave);
                //Descriptor 要重新定义不能继续用cv::Mat
                desc.x = scale_factor*(kp.pt.x + 0.5f) - 0.5f;
                desc.y = scale_factor*(kp.pt.y + 0.5f) - 0.5f;
                desc.scale = this->keypoint_absolute_scale(kp);
                desc.orientation = orientations[j];
                if(this->descriptor_assignment(kp, desc, octave)) {
                    this->descriptors.push_back(desc.data);
                    //put descriptors into mat
                    Sift_KeyPoint skp;
                    skp.pt = cv::Point2f(desc.x,desc.y);
                    skp.sample = kp.sample;
                    skp.angle = kp.angle;
                    skp.scale = desc.scale;
                    new_keypoints.push_back(skp);
                }
            }
        }
        keypoints.clear();
        for(size_t i=0; i<new_keypoints.size(); i++){
            keypoints.push_back(new_keypoints[i]);
        }
    }

    void Desc_s128::generate_grad_ori_images(suo15features::Simple_Octave *octave) {
        octave->grad.clear();
        octave->grad.reserve(octave->img.size());
        octave->ori.clear();
        octave->ori.reserve(octave->img.size());

        const int width = octave->img[0].cols;
        const int height = octave->img[0].rows;

        for(std::size_t i=0; i<octave->img.size(); i++){
            const cv::Mat img = octave->img[i];
            cv::Mat grad(height, width, CV_32FC1);
            cv::Mat ori(height, width, CV_32FC1);

            int image_iter = width + 1;
            for(int y=1; y<height-1; ++y, image_iter += 2)
                for(int x = 1; x <width -1; ++x, ++image_iter){
                float m1x = img.at<float>(image_iter - 1);
                float p1x = img.at<float>(image_iter + 1);
                float m1y = img.at<float>(image_iter - width);
                float p1y = img.at<float>(image_iter + width);
                float dx = 0.5f* (p1x - m1x);
                float dy = 0.5f* (p1y - m1y);

                float atan2f = atan2(dy, dx);
                grad.at<float>(image_iter) = sqrt(dx*dx + dy*dy);
                ori.at<float>(image_iter) = atan2f<0.0f
                        ? atan2f + MATH_PI * 2.0f : atan2f;
            }
            octave->grad.push_back(grad);
            octave->ori.push_back(ori);
        }
    }

    void Desc_s128::orientation_assignment(const suo15features::Sift_KeyPoint &kp,
                                           const suo15features::Simple_Octave *octave,
                                           vector<float> &orientations) {

        const int nbins = 36;
        const float nbinsf = static_cast<float>(nbins);

        float hist[nbins];
        fill(hist, hist+nbins, 0.0f);

        const int ix = static_cast<int>(kp.pt.x + 0.5f);
        const int iy = static_cast<int>(kp.pt.y + 0.5f);
        const int is = static_cast<int>(round(kp.sample));
        const float sigma = this->keypoint_relative_scale(kp);

        cv::Mat grad = octave->grad[is+1];
        cv::Mat ori = octave->ori[is+1];

        const int width = grad.cols;
        const int height = grad.rows;

        /*
        * Compute window size 'win', the full window has  2 * win + 1  pixel.
        * The factor 3 makes the window large enough such that the gaussian
        * has very little weight beyond the window. The value 1.5 is from
        * the SIFT paper. If the window goes beyond the image boundaries,
        * the keypoint is discarded.
        */
        const float sigma_factor = 1.5f;
        int win = static_cast<int>(sigma * sigma_factor * 3.0f);
        if(ix < win || ix + win >= width || iy < win || iy + win >= height)
            return;
        /* Center of keypoint index. */
        int center = iy * width + ix;
        float const dxf = kp.pt.x - static_cast<float>(ix);
        float const dyf = kp.pt.y - static_cast<float>(iy);
        float const maxdist = static_cast<float>(win*win) + 0.5f;

        /* Populate histogram over window, intersected with (1,1), (w-2,h-2). */
        for (int dy = -win; dy <= win; ++dy)
        {
            int const yoff = dy * width;
            for (int dx = -win; dx <= win; ++dx)
            {
                /* Limit to circular window (centered at accurate keypoint). */
                const float dist = pow((dx-dxf), 2) + pow((dy-dyf), 2);
                if (dist > maxdist)
                    continue;

                float gm = grad.at<float>(center + yoff + dx); // gradient magnitude
                float go = ori.at<float>(center + yoff + dx); // gradient orientation
                float weight = gaussian_xx(dist, sigma * sigma_factor);
                int bin = static_cast<int>(nbinsf * go / (2.0f * MATH_PI));
                bin = clamp(bin, 0, nbins - 1);
                hist[bin] += gm * weight;
            }
        }

        for(int i=0; i<6; i++){
            float first = hist[0];
            float prev = hist[nbins - 1];
            for(int j=0; j<nbins-1; ++j){
                float current = hist[j];
                hist[j] = (prev+current+hist[j+1])/3.0f;
                prev = current;
            }
            hist[nbins -1] = (prev + hist[nbins-1]+first)/3.0f;
        }

        float maxh = *max_element(hist, hist + nbins);

        for(int i=0; i<nbins; ++i){
            float h0 = hist[(i+nbins-1)%nbins];
            float h1 = hist[i];
            float h2 = hist[(i+1)%nbins];

            if(h1 <= 0.8f*maxh || h1 <= h0 || h1 <= h2)
                continue;

            float x = -0.5f*(h2-h0)/(h0 - 2.0f*h1 + h2);
            float o = 2.0f* MATH_PI * (x+(float)i + 0.5f)/nbinsf;
            orientations.push_back(o);
        }
    }

    bool Desc_s128::descriptor_assignment(const suo15features::Sift_KeyPoint &kp, suo15features::Descriptor &desc,
                                          const suo15features::Simple_Octave *octave) {
        const int PXB = 4;
        const int OHB = 8;

        const int ix = static_cast<int>(kp.pt.x+0.5f);
        const int iy = static_cast<int>(kp.pt.y+0.5f);
        const int is = static_cast<int>(round(kp.sample));

        const float dxf = kp.pt.x - static_cast<float>(ix);
        const float dyf = kp.pt.y - static_cast<float>(iy);
        const float sigma = this->keypoint_relative_scale(kp);

        cv::Mat grad = octave->grad[is+1];
        cv::Mat ori = octave->ori[is+1];
        const int width = grad.cols;
        const int height = grad.rows;
        //描述子要在不同的宽度和长度的状态下
        desc.data.create(1, 128, CV_32FC1);
        //desc.data.resize(128);
        const float sino = sin(desc.orientation);
        const float coso = cos(desc.orientation);

        const float binsize = 3.0f * sigma;
        int win = MATH_SQRT2 * binsize * (float)(PXB + 1) * 0.5f;
        if(ix < win || ix + win >= width || iy < win || iy + win >= height)
            return false;
        int const center = iy * width + ix; // Center pixel at KP location
        for (int dy = -win; dy <= win; ++dy)
        {
            int const yoff = dy * width;
            for (int dx = -win; dx <= win; ++dx)
            {
                /* Get pixel gradient magnitude and orientation. */
                float const mod = grad.at<float>(center + yoff + dx);
                float const angle = ori.at<float>(center + yoff + dx);
                float theta = angle - desc.orientation;
                if (theta < 0.0f)
                    theta += 2.0f * MATH_PI;

                /* Compute fractional coordinates w.r.t. the window. */
                float const winx = (float)dx - dxf;
                float const winy = (float)dy - dyf;

                /*
                 * Compute normalized coordinates w.r.t. bins. The window
                 * coordinates are rotated around the keypoint. The bins are
                 * chosen such that 0 is the coordinate of the first bins center
                 * in each dimension. In other words, (0,0,0) is the coordinate
                 * of the first bin center in the three dimensional histogram.
                 */
                float binoff = (float)(PXB - 1) / 2.0f;
                float binx = (coso * winx + sino * winy) / binsize + binoff;
                float biny = (-sino * winx + coso * winy) / binsize + binoff;
                float bint = theta * (float)OHB / (2.0f * MATH_PI) - 0.5f;

                /* Compute circular window weight for the sample. */
                float gaussian_sigma = 0.5f * (float)PXB;
                float gaussian_weight = gaussian_xx
                        (pow((binx - binoff), 2) + pow((biny - binoff), 2),
                         gaussian_sigma);

                /* Total contribution of the sample in the histogram is now: */
                float contrib = mod * gaussian_weight;

                /*
                 * Distribute values into bins (using trilinear interpolation).
                 * Each sample is inserted into 8 bins. Some of these bins may
                 * not exist, because the sample is outside the keypoint window.
                 */
                int bxi[2] = { (int)std::floor(binx), (int)std::floor(binx) + 1 };
                int byi[2] = { (int)std::floor(biny), (int)std::floor(biny) + 1 };
                int bti[2] = { (int)std::floor(bint), (int)std::floor(bint) + 1 };

                float weights[3][2] = {
                        { (float)bxi[1] - binx, 1.0f - ((float)bxi[1] - binx) },
                        { (float)byi[1] - biny, 1.0f - ((float)byi[1] - biny) },
                        { (float)bti[1] - bint, 1.0f - ((float)bti[1] - bint) }
                };

                // Wrap around orientation histogram
                if (bti[0] < 0)
                    bti[0] += OHB;
                if (bti[1] >= OHB)
                    bti[1] -= OHB;

                /* Iterate the 8 bins and add weighted contrib to each. */
                int const xstride = OHB;
                int const ystride = OHB * PXB;
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x)
                        for (int t = 0; t < 2; ++t)
                        {
                            if (bxi[x] < 0 || bxi[x] >= PXB
                                || byi[y] < 0 || byi[y] >= PXB)
                                continue;

                            int idx = bti[t] + bxi[x] * xstride + byi[y] * ystride;
                            desc.data.at<float>(idx) += contrib * weights[0][x]
                                              * weights[1][y] * weights[2][t];
                        }
            }
        }

        /* Normalize the feature vector. */
        //存在在计算的时候Mat是0请款？？这样进行正则化会出问题！！！
        cv::normalize(desc.data, desc.data);
        //desc.data.normalize();
        /* Truncate descriptor values to 0.2. */
        for (int i = 0; i < PXB * PXB * OHB; ++i)
            desc.data.at<float>(i) = std::min(desc.data.at<float>(i), 0.2f);

        /* Normalize once again. */
        cv::normalize(desc.data, desc.data);
        //desc.data.normalize();
        return true;
    }

    Simple_Octaves Desc_s128::create_octaves(const cv::Mat &image) {
        this->octaves.clear();
        //image 进行数据转换，将数据转换成float类型
        set_image(image);
        if(this->desc_s128_options.min_octave < 0){
            float scale = 0.5;//放大尺寸的，以小于0的初值来说
            cv::Size sz(cvRound((float)image.cols/scale), cvRound((float)image.rows/scale));
            cv::Mat img;
            cv::resize(image, img, sz);
            //尺寸放大一倍，不进行高斯模糊的原图
            //  the much important function!!! */
            this->add_octave(img, this->desc_s128_options.inherent_blur_sigma*2.0f,
                             this->desc_s128_options.base_blur_sigma);
        }

        cv::Mat img = this->orig;
        for(int i=0; i<this->desc_s128_options.min_octave; ++i){
            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            cv::resize(img, img, sz);//并没有运行！！！
        }//降采样一半, 通过这里的操作img变成了，min_octave 一般是0，要做上采样则是-1
        //Gauss_Blur GB;
        float img_sigma = this->desc_s128_options.inherent_blur_sigma;
        for(int i = max(0, this->desc_s128_options.min_octave);
            i <= this->desc_s128_options.max_octave; ++i){
            this->add_octave(img, img_sigma, this->desc_s128_options.base_blur_sigma);

            cv::Size sz(cvRound((float)img.cols/2), cvRound((float)img.rows/2));
            cv::Mat pre_base = octaves[octaves.size()-1].img[0];
            cv::resize(pre_base, img, sz);
            img_sigma = this->desc_s128_options.base_blur_sigma;
        }
        return this->octaves;
    }

    void Desc_s128::set_image (const cv::Mat& img)
    {
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
    }

    void Desc_s128::add_octave(cv::Mat &image, float has_sigma, float target_sigma)
    {
        float sigma = std::sqrt(pow(target_sigma, 2) - pow(has_sigma, 2));
        /*std::cout << "Pre-blurring image to sigma " << target_sigma << " (has "
            << has_sigma << ", blur = " << sigma << ")..." << std::endl;*/
        int ks = ceil(sigma*2.884f);
        cv::Size sz(2*ks+1, 2*ks+1);

        cv::Mat base;
        if(target_sigma > has_sigma){
            cv::GaussianBlur(image, base, sz, sigma, sigma, cv::BORDER_REPLICATE);
        } else
            base = image.clone();

        /* Create the new octave and add initial image. */
        this->octaves.push_back(Simple_Octave());
        Simple_Octave& oct = this->octaves.back();
        oct.img.push_back(base);

        /* 'k' is the constant factor between the scales in scale space. */
        float const k = std::pow(2.0f, 1.0f / this->desc_s128_options.num_samples_per_octave);
        sigma = target_sigma;

        /* Create other (s+2) samples of the octave to get a total of (s+3). */
        for (int i = 1; i < this->desc_s128_options.num_samples_per_octave + 3; ++i)
        {
            /* Calculate the blur sigma the image will get. */
            float sigmak = sigma * k;
            float blur_sigma = std::sqrt(pow(sigmak, 2) - pow(sigma, 2));
            /* Blur the image to create a new scale space sample. */
            //std::cout << "Blurring image to sigma " << sigmak << " (has " << sigma
            //    << ", blur = " << blur_sigma << ")..." << std::endl;
            int cks = ceil(blur_sigma*2.884f);
            cv::Size csz(2*cks+1, 2*cks+1);
            cv::Mat img;
            cv::GaussianBlur(base, img, csz, blur_sigma, blur_sigma, cv::BORDER_REPLICATE);
            oct.img.push_back(img);
            /* Update previous image and sigma for next round. */
            base = img;
            sigma = sigmak;
        }
    }

    float Desc_s128::keypoint_relative_scale(const suo15features::Sift_KeyPoint &kp) {
        return this->desc_s128_options.base_blur_sigma * std::pow(2.0f,
                   (kp.sample + 1.0f) / this->desc_s128_options.num_samples_per_octave);
    }

    float Desc_s128::keypoint_absolute_scale(const suo15features::Sift_KeyPoint &kp) {
        return this->desc_s128_options.base_blur_sigma * std::pow(2.0f,
                   kp.octave + (kp.sample + 1.0f) / this->desc_s128_options.num_samples_per_octave);
    }
}