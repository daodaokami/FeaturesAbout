//
// Created by lut on 18-10-12.
//

#ifndef LUT15VO_COMMON_INCLUDE_H
#define LUT15VO_COMMON_INCLUDE_H
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SE3;
using Sophus::SO3;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using cv::Mat;

#include <vector>
#include <list>
#include <memory>
#include <string>
#include <set>
#include <iostream>
#include <unordered_map>
#include <map>
using namespace std;

#define MATH_EPSILON_EQ(x,v,eps) (((v - eps) <= x) && (x <= (v + eps)))
#endif //LUT15VO_COMMON_INCLUDE_H
