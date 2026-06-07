/**
 * @file    KalmanFilter.cpp
 * @brief   EKF 单目标跟踪器实现
 */

#include "KalmanFilter.h"

// ============================================================
//  静态成员
// ============================================================
int KalmanFilter::next_id_ = 0;

// ============================================================
//  构造 & 初始化
// ============================================================
KalmanFilter::KalmanFilter()
    : id_(next_id_++)
    , miss_count_(0)
{
    buildMatrices();
}

void KalmanFilter::init(float x, float y, float w, float h)
{
    X_ = cv::Mat::zeros(8, 1, CV_32F);
    X_.at<float>(0) = x;
    X_.at<float>(1) = y;
    X_.at<float>(2) = w;
    X_.at<float>(3) = h;
    // 速度初始为 0

    P_ = cv::Mat::eye(8, 8, CV_32F) * 1000.f;  // 初始不确定性较大
    miss_count_ = 0;
}

// ============================================================
//  buildMatrices — 构建恒定速度模型矩阵
// ============================================================
void KalmanFilter::buildMatrices()
{
    // ---- F: 状态转移矩阵 (8x8) ----
    // 恒定速度模型: x' = x + vx, vx' = vx
    //   [1 0 0 0 1 0 0 0]
    //   [0 1 0 0 0 1 0 0]
    //   [0 0 1 0 0 0 1 0]
    //   [0 0 0 1 0 0 0 1]
    //   [0 0 0 0 1 0 0 0]
    //   [0 0 0 0 0 1 0 0]
    //   [0 0 0 0 0 0 1 0]
    //   [0 0 0 0 0 0 0 1]
    F_ = cv::Mat::eye(8, 8, CV_32F);
    F_.at<float>(0, 4) = 1.f;  // x  += vx
    F_.at<float>(1, 5) = 1.f;  // y  += vy
    F_.at<float>(2, 6) = 1.f;  // w  += vw
    F_.at<float>(3, 7) = 1.f;  // h  += vh

    // ---- H: 观测矩阵 (4x8) ----
    // 只观测位置和尺寸, 不观测速度
    H_ = cv::Mat::zeros(4, 8, CV_32F);
    H_.at<float>(0, 0) = 1.f;
    H_.at<float>(1, 1) = 1.f;
    H_.at<float>(2, 2) = 1.f;
    H_.at<float>(3, 3) = 1.f;

    // ---- Q: 过程噪声协方差 (8x8) ----
    // 主要不确定性在速度分量上
    Q_ = cv::Mat::eye(8, 8, CV_32F) * 0.01f;
    // 位置噪声
    Q_.at<float>(0, 0) = 1.f;
    Q_.at<float>(1, 1) = 1.f;
    Q_.at<float>(2, 2) = 1.f;
    Q_.at<float>(3, 3) = 1.f;
    // 速度噪声较大 (运动可能变化)
    Q_.at<float>(4, 4) = 50.f;
    Q_.at<float>(5, 5) = 50.f;
    Q_.at<float>(6, 6) = 10.f;
    Q_.at<float>(7, 7) = 10.f;

    // ---- R: 观测噪声协方差 (4x4) ----
    // 检测器有一定噪声, 中心比尺寸更稳定
    R_ = cv::Mat::eye(4, 4, CV_32F) * 10.f;
    R_.at<float>(0, 0) = 25.f;   // x 噪声
    R_.at<float>(1, 1) = 25.f;   // y 噪声
    R_.at<float>(2, 2) = 50.f;   // w 噪声 (尺寸估计更不准确)
    R_.at<float>(3, 3) = 50.f;   // h 噪声
}

// ============================================================
//  computeJacobian
// ============================================================
cv::Mat KalmanFilter::computeJacobian() const
{
    // 在恒定速度模型中, 状态转移是线性的 → 雅可比即 F 矩阵
    // 保留此方法以便未来引入非线性模型
    return F_.clone();
}

// ============================================================
//  predict — 时间更新
// ============================================================
cv::Vec4f KalmanFilter::predict()
{
    // 1) 状态预测: X' = F * X
    X_ = F_ * X_;

    // 2) 协方差预测: P' = F * P * F^T + Q
    cv::Mat F_jac = computeJacobian();
    P_ = F_jac * P_ * F_jac.t() + Q_;

    return state();
}

// ============================================================
//  update — 测量更新 (标准卡尔曼更新)
// ============================================================
cv::Vec4f KalmanFilter::update(float x, float y, float w, float h)
{
    // 1) 观测向量 Z (4x1)
    cv::Mat Z = (cv::Mat_<float>(4, 1) << x, y, w, h);

    // 2) 观测残差: y = Z - H * X
    cv::Mat y_res = Z - H_ * X_;

    // 3) 残差协方差: S = H * P * H^T + R
    cv::Mat S = H_ * P_ * H_.t() + R_;

    // 4) 卡尔曼增益: K = P * H^T * S^{-1}
    cv::Mat K = P_ * H_.t() * S.inv();

    // 5) 状态更新: X = X + K * y
    X_ = X_ + K * y_res;

    // 6) 协方差更新: P = (I - K * H) * P
    cv::Mat I = cv::Mat::eye(8, 8, CV_32F);
    P_ = (I - K * H_) * P_;

    resetMiss();
    return state();
}

// ============================================================
//  state / velocity 访问器
// ============================================================
cv::Vec4f KalmanFilter::state() const
{
    return cv::Vec4f(
        X_.at<float>(0), X_.at<float>(1),
        X_.at<float>(2), X_.at<float>(3));
}

cv::Vec4f KalmanFilter::velocity() const
{
    return cv::Vec4f(
        X_.at<float>(4), X_.at<float>(5),
        X_.at<float>(6), X_.at<float>(7));
}
