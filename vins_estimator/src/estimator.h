#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    void setParameter(); // 设置参数，初始化 Estimator 参数

    // interface（接口函数）
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity); // 处理IMU数据
    // map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image
    // int: Feature ID, int: Camera ID, <double, 7, 1> x,y,z,u,v,vx,vy
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header); // 处理图像数据
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r); // 设置重定位帧信息

    // internal
    void clearState(); // 清除状态，初始化状态
    bool initialStructure(); // 初步构建特征点三角化结构
    bool visualInitialAlign(); // 视觉初始对齐
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l); // 计算相对位姿
    void slideWindow(); // 滑动窗口
    void solveOdometry(); // 解算里程计
    void slideWindowNew(); // 滑动窗口（新帧）
    void slideWindowOld(); // 滑动窗口（旧帧）
    void optimization(); // 优化
    void vector2double(); // 向量转换为双精度数组
    void double2vector(); // 双精度数组转换为向量
    bool failureDetection(); // 失败检测


    enum SolverFlag // 求解器标志
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag // 边缘化标志
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g; // 重力加速度
    MatrixXd Ap[2], backup_A; // 优化问题的雅可比矩阵
    VectorXd bp[2], backup_b; // 优化问题的残差向量

    Matrix3d ric[NUM_OF_CAM]; // 相机到IMU的外参旋转矩阵
    Vector3d tic[NUM_OF_CAM]; // 相机到IMU的外参平移向量

    // 滑窗PVQB
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager; // 特征点管理器
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
