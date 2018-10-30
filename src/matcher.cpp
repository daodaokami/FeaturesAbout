//
// Created by lut on 18-10-12.
//

#include <nearest_neighbor.h>
#include "../include/matcher.h"

namespace suo15features{
    vector<CP> Matcher::GetMatchedKeypoints(const vector<cv::KeyPoint> &keypoints_1,
        const cv::Mat &descriptors_1,
        const vector<cv::KeyPoint> &keypoints_2,
        const cv::Mat &descriptors_2) {

        cout<<"father matcher!"<<endl;
        return vector<pair<size_t, size_t>>();
    }

    void Matcher::oneway_match(const suo15features::Matcher_options &options,
                               const Descriptors& set_1, int set_1_size,
                               const Descriptors& set_2, int set_2_size,
                               vector<int>& result) {
        result.clear();
        result.resize(set_1_size, -1);

        if(set_1_size == 0 || set_2_size == 0)
            return;
        float const square_dist_thres = pow(options.distance_threshold, 2);
        //最近邻要进行修改！！
        NearestNeighbor nn;

        //nn.set_elements(set_2);
        //nn.set_element_dimensions(set_2_size);
        //nn.set_element_dimensions(options.descriptor_length);
    }

    void Matcher::twoway_match(const suo15features::Matcher_options &options,
                  const suo15features::Descriptors &set_1,
                  size_t set_1_size,
                  const suo15features::Descriptors &set_2, size_t set_2_size,
                  suo15features::Result &matches) {


    }
};