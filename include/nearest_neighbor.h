//
// Created by lut on 18-10-15.
//

#ifndef LUT15VO_NEAREST_NEIGHBOR_H
#define LUT15VO_NEAREST_NEIGHBOR_H

#include "common_include.h"

namespace suo15features {
    template <typename T>
    class NearestNeighbor{
    public:
        struct Result{
            T dist_1st_best;
            T dist_2nd_best;
            int index_1st_best;
            int index_2nd_best;
        };

    public:
        NearestNeighbor(void);

        void set_elements(T const* elements);
        void set_element_dimensions(int element_dimensions);

        void set_num_elements(int num_elements);
        void find(T const* query, Result* result) const;

        int get_element_dimensions(void) const;

    private:
        int dimensions;
        int num_elements;
        T const* elements;
    };

    inline NearestNeighbor<T>::NearestNeighbor(void):
            dimensions(64), num_elements(0), elements(nullptr){}

    template <typename T>
    inline void
    NearestNeighbor<T>::set_elements (T const* elements)
    {
        this->elements = elements;
    }

    template <typename T>
    inline void
    NearestNeighbor<T>::set_element_dimensions (int element_dimensions)
    {
        this->dimensions = element_dimensions;
    }

    template <typename T>
    inline void
    NearestNeighbor<T>::set_num_elements (int num_elements)
    {
        this->num_elements = num_elements;
    }

    template <typename T>
    inline int
    NearestNeighbor<T>::get_element_dimensions (void) const
    {
        return this->dimensions;
    }
}


#endif //LUT15VO_NEAREST_NEIGHBOR_H
