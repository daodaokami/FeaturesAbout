//
// Created by lut on 18-10-15.
//

#ifndef LUT15VO_NEAREST_NEIGHBOR_H
#define LUT15VO_NEAREST_NEIGHBOR_H

#include "common_include.h"

namespace suo15features {

    class NearestNeighbor{
    public:
        struct Result{
            float dist_1st_best;
            float dist_2nd_best;
            int index_1st_best;
            int index_2nd_best;
        };
    public:
        NearestNeighbor (void);

        void set_elements(const cv::Mat& elements);
        void set_element_dimensions(int element_dimensions);

        void set_num_elements(int num_elements);
        void find(const cv::Mat& query, Result* result) const;

        int get_element_dimensions(void) const;

    private:
        int dimensions;
        int num_elements;
        cv::Mat elements;
    };

    inline NearestNeighbor::NearestNeighbor(void):
            dimensions(64),
            num_elements(0){
        elements.create(1, dimensions, CV_32FC1);
    }

    inline void
    NearestNeighbor::set_elements (const cv::Mat& elements)
    {
        this->elements = elements;
    }

    inline void
    NearestNeighbor::set_element_dimensions (int element_dimensions)
    {
        this->dimensions = element_dimensions;
    }

    inline void
    NearestNeighbor::set_num_elements (int num_elements)
    {
        this->num_elements = num_elements;
    }

    inline int
    NearestNeighbor::get_element_dimensions (void) const
    {
        return this->dimensions;
    }
}

#endif //LUT15VO_NEAREST_NEIGHBOR_H
