add_library(suo15features SHARED
        accum_conv.cpp
        descry.cpp
        desc_s128.cpp
        desc_s64.cpp
        desc_b256.cpp
        detector.cpp
        sift_keypoint.cpp
        gauss_blur.cpp
        detector_fast.cpp
        detector_sift.cpp
        detector_surf.cpp
        detector_orb.cpp
        matcher.cpp
        matcher_bf.cpp
        matcher_knn.cpp
        feature_set.cpp
        visualize.cpp
        matcher_stereo.cpp
        datas_map.cpp
        spacenear_matcher.cpp
        spatical_subdivision.cpp
        )

target_link_libraries(
        suo15features ${THIRD_PARTY_LIBS}
)