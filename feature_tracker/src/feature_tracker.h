#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time); // 读取图像并记录当前时间

    void setMask(); // 设置图像掩码，用于在特征点匹配过程中限制搜索范围

    void addPoints(); // 向图像中添加新的特征点

    bool updateID(unsigned int i); // 更新特征点的ID号

    void readIntrinsicParameter(const string &calib_file); // 从文件中读取相机的内部参数

    void showUndistortion(const string &name); // 展示图像的畸变校正效果

    void rejectWithF(); // 使用基于基础矩阵的几何约束（F矩阵）来拒绝错误的匹配点

    void undistortedPoints(); // 对图像中的点进行畸变校正

    cv::Mat mask; // 图像掩码
    cv::Mat fisheye_mask; // 鱼眼相机畸变校正掩码
    cv::Mat prev_img, cur_img, forw_img; // 上一帧图像、当前帧图像、下一帧图像
    vector<cv::Point2f> n_pts; // 新检测到的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts; // 上一帧、当前帧、下一帧的特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts; // 上一帧、当前帧的畸变校正后的特征点
    vector<cv::Point2f> pts_velocity; // 特征点的速度
    vector<int> ids; // 特征点的ID号
    vector<int> track_cnt; // 特征点的追踪次数
    map<int, cv::Point2f> cur_un_pts_map; // 当前帧的畸变校正后的特征点映射
    map<int, cv::Point2f> prev_un_pts_map; // 上一帧的畸变校正后的特征点映射
    camodocal::CameraPtr m_camera; // 相机内部参数模型
    double cur_time; // 当前时间
    double prev_time; // 上一帧时间

    static int n_id; // 静态变量，用于记录特征点的ID号
};
