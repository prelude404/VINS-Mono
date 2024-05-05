#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function; // 核函数
    std::vector<double *> parameter_blocks; // 优化变量
    std::vector<int> drop_set; // 待marg变量的序号

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; // Jacobian
    Eigen::VectorXd residuals; // 误差项

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 管理边缘化
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); // 添加残差块信息
    void preMarginalize(); // 预处理边缘化信息
    void marginalize(); // 边缘化
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    int m, n; // m：需要marg掉的变量的总维度；n：需要保留的变量的总维度
    // long: 每个变量的数据地址
    // m+n：所有与约束相关变量的维度总和（即H矩阵维度）
    // m：1(M0)*9(v+b) + 1(P0)*6(p+q) + 1(Tbc)*6 + 6(/lambda)*1
    std::unordered_map<long, int> parameter_block_size; // 每个变量的维度
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; // 每个变量在H矩阵的索引
    std::unordered_map<long, double *> parameter_block_data; // 每个变量的数据

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
