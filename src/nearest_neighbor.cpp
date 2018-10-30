//
// Created by lut on 18-10-15.
//

#include <algorithm>
#include <iostream>

#include "nearest_neighbor.h"

namespace suo15features{

    void
    NearestNeighbor::find (const cv::Mat& query,
                                  NearestNeighbor::Result* result) const
    {
        /* Result distances are shamelessly misused to store inner products. */
        result->dist_1st_best = 0.0f;
        result->dist_2nd_best = 0.0f;
        result->index_1st_best = 0;
        result->index_2nd_best = 0;

        /*float_inner_prod(query, result, this->elements,
                         this->num_elements, this->dimensions);*/

        /*
         * Compute actual (square) distances.
         */
        result->dist_1st_best = std::max(0.0f, 2.0f - 2.0f * result->dist_1st_best);
        result->dist_2nd_best = std::max(0.0f, 2.0f - 2.0f * result->dist_2nd_best);
    }
}