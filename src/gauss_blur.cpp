//
// Created by lut on 18-10-19.
//

#include "gauss_blur.h"

namespace suo15features{
    cv::Mat Gauss_Blur::rescale_double_size_supersample(const cv::Mat &img) {
        cv::Size sz(img.cols, img.rows);
        cv::Mat out;
        cv::resize(img, out, sz);
        return out;
    }

    cv::Mat Gauss_Blur::rescale_half_size_gaussian(const cv::Mat &img, float sigma) {
       if(img.channels() != 1)
           throw invalid_argument("Invalid input img channels");
       const int iw = img.cols;
       const int ih = img.rows;
       const int ic = img.channels();

       const int ow = (iw+1) >> 1;
       const int oh = (ih+1) >> 1;

        if(iw < 2 || ih < 2)
            throw invalid_argument("Invalid input img size");
        cv::Mat out;
        out.create(oh, ow, CV_32FC1);
        const float w1 = exp(-0.5f/(2.0f*sigma*sigma));
        const float w2 = exp(-2.5f/(2.0f*sigma*sigma));
        const float w3 = exp(-4.5f/(2.0f*sigma*sigma));

        int outpos = 0;
        int const rowstride = iw*ic;//相当于宽度

        for(int y=0; y<oh; ++y){
            int y2 = (int)y<<1;
            const float* row[4];
            int pos0 = max(0, y2-1)*rowstride;
            int pos1 = (y2*rowstride);
            int pos2 = min((int)ih-1, y2+1)*rowstride;
            int pos3 = min((int)ih-1, y2+2)*rowstride;

            row[0] = &img.at<float>(pos0/iw, pos0%iw);
            row[1] = &img.at<float>(pos1/iw, pos1%iw);
            row[2] = &img.at<float>(pos2/iw, pos2%iw);
            row[3] = &img.at<float>(pos3/iw, pos3%iw);

            for(int x = 0; x<ow; ++x){
                int x2 = (int)x<<1;
                int xi[4];
                xi[0] = max(0, x2-1)*ic ;
                xi[1] = x2*ic;
                xi[2] = min((int)iw -1, x2 + 1)*ic;
                xi[3] = min((int)iw -1, x2 + 2)*ic;
                for(int c = 0; c<ic; ++c){
                    Accum_Conv accum(0.0f);
                    accum.add(row[0][xi[0] + c], w3);
                    accum.add(row[0][xi[1] + c], w2);
                    accum.add(row[0][xi[2] + c], w2);
                    accum.add(row[0][xi[3] + c], w3);

                    accum.add(row[1][xi[0] + c], w2);
                    accum.add(row[1][xi[1] + c], w1);
                    accum.add(row[1][xi[2] + c], w1);
                    accum.add(row[1][xi[3] + c], w2);

                    accum.add(row[2][xi[0] + c], w2);
                    accum.add(row[2][xi[1] + c], w1);
                    accum.add(row[2][xi[2] + c], w1);
                    accum.add(row[2][xi[3] + c], w2);

                    accum.add(row[3][xi[0] + c], w3);
                    accum.add(row[3][xi[1] + c], w2);
                    accum.add(row[3][xi[2] + c], w2);
                    accum.add(row[3][xi[3] + c], w3);

                    int rw = outpos/ow, cl = outpos%ow;
                    out.at<float>(rw, cl) = accum.normalized();
                    ++outpos;
                }
            }
        }
        return out;
    }
}