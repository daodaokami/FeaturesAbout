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

namespace suo15features {
    enum FeatureTypes {
        FEATURE_SIFT = 0,
        FEATURE_ORB = 1,
        FEATURE_FAST = 2,
        FEATURE_SURF = 3,
        FEATURE_ALL = 0xFF
    };

    struct Options {
        Options(void);

        FeatureTypes feature_types;
        SIFT_options sift_opts;
        ORB_options orb_opts;
        Fast_options fast_opts;
        //SURF_options surf_opts;
    };

    class FeatureSet {
    public:
        FeatureSet();
        explicit FeatureSet(const Options& options);
        void set_options(const Options& options);

        //todo compute features
        /* compute the features specified by the options */
        void compute_features(const cv::Mat& image);

        void normalize_feature_position();

        void clear_descriptors();

    public:
        int width, height;
        vector<cv::Point2f> positions;
        vector<unsigned char> colors;

        cv::Mat descriptors;
        Descriptors sift_descriptors;
    private:
        Options opts;

        void compute_sift(const cv::Mat& image);
    };

}

#endif //LUT15VO_FEATURE_SET_H
