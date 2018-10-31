//
// Created by lut on 18-10-26.
//

#ifndef LUT15VO_FEATURE_SET_H
#define LUT15VO_FEATURE_SET_H

#include "common_include.h"
#include "detector_sift.h"
#include "detector_surf.h"
#include "detector_fast.h"
#include "detector_orb.h"
#include "desc_s64.h"
#include "desc_s128.h"
#include "desc_b256.h"

namespace suo15features {
    enum FeatureTypes {
        FEATURE_SIFT = 0,
        FEATURE_ORB = 1,
        FEATURE_FAST = 2,
        FEATURE_SURF = 3,
        FEATURE_ALL = 0xFF
    };

    struct Feature_options {
        Feature_options(void);
        Feature_options(const FeatureTypes& featureTypes,
                const SIFT_options& sift_options,
                const ORB_options& orb_options,
                const Fast_options& fast_options);
        FeatureTypes feature_types;
        SIFT_options sift_opts;
        ORB_options orb_opts;
        Fast_options fast_opts;
        //SURF_options surf_opts;

    };

    Feature_options::Feature_options() {}
    Feature_options::Feature_options(const FeatureTypes& featureTypes,
                                     const SIFT_options& sift_options,
                                     const ORB_options& orb_options,
                                     const Fast_options& fast_options){
        this->feature_types = featureTypes;
        this->sift_opts = sift_options;
        this->orb_opts = orb_options;
        this->fast_opts = fast_options;
    }

    class FeatureSet {
    public:
        FeatureSet();
        explicit FeatureSet(const Feature_options& options);
        void set_options(const Feature_options& options);

        //todo compute features
        /* compute the features specified by the options */
        void compute_features(const cv::Mat& image);

        void normalize_feature_position();

        void clear_descriptors();

    public:
        int width, height;
        vector<cv::Point2f> positions;
        vector<unsigned char> colors;
        //主要会划分两种的Descriptor，一种是直接的Mat类型, 一种自带坐标
        cv::Mat descriptors;
        Descriptors sift_descriptors;
    private:
        Feature_options feature_options;

        void compute_sift(const cv::Mat& image);
        void compute_orb(const cv::Mat& image);
    };

    template<typename T>
    static bool compare_scale(const T& desc1, const T& desc2){
        return desc1.scale > desc2.scale;
    }

    inline FeatureSet::FeatureSet() {}

    inline FeatureSet::FeatureSet(const suo15features::Feature_options &options):
            feature_options(options) {}

    inline void FeatureSet::set_options(const suo15features::Feature_options &options) {
        this->feature_options = options;
    }
}

#endif //LUT15VO_FEATURE_SET_H
