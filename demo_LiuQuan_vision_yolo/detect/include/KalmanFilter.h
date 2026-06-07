/**
 * @file    KalmanFilter.h
 * @brief   扩展卡尔曼滤波器 (EKF) — 用于单目标跟踪
 *
 * 状态向量 (8 维):  [x, y, w, h, vx, vy, vw, vh]
 *   - x, y:      边界框中心坐标 (像素)
 *   - w, h:      边界框宽高 (像素)
 *   - vx, vy:    中心速度 (像素/帧)
 *   - vw, vh:    尺寸变化率 (像素/帧)
 *
 * 观测向量 (4 维): [x, y, w, h]  — 来自 YOLO 检测
 *
 * 运动模型: 匀速直线运动 (恒定速度, dt=1)
 * 非线性: 无 (本问题中状态转移是线性的, EKF 退化为 KF)
 *         保留 EKF 接口以支持未来扩展 (如引入非线性观测)
 */
#pragma once

#include <opencv2/core.hpp>

// ============================================================
//  EKF 单目标跟踪器
// ============================================================
class KalmanFilter {
public:
    /// 构造函数
    KalmanFilter();

    /**
     * @brief 初始化滤波器状态
     * @param x, y, w, h  初始边界框
     */
    void init(float x, float y, float w, float h);

    /**
     * @brief  预测步骤 (时间更新)
     * @return 预测后的状态 [x, y, w, h]
     */
    cv::Vec4f predict();

    /**
     * @brief  更新步骤 (测量更新)
     * @param  x, y, w, h  观测值 (检测结果)
     * @return 更新后的状态 [x, y, w, h]
     */
    cv::Vec4f update(float x, float y, float w, float h);

    /// 获取当前状态估计
    cv::Vec4f state() const;

    /// 获取当前速度估计
    cv::Vec4f velocity() const;

    /// 获取状态协方差矩阵 (调试用)
    const cv::Mat& covariance() const { return P_; }

    /// 获取跟踪器 ID
    int id() const { return id_; }

    /// 丢失计数
    int  missCount() const { return miss_count_; }
    void incrementMiss()    { miss_count_++; }
    void resetMiss()        { miss_count_ = 0; }

private:
    static int next_id_;    // 全局 ID 分配器

    int     id_;            // 跟踪器 ID
    int     miss_count_;    // 连续未匹配帧数

    // 状态 X (8x1)
    cv::Mat X_;   // [x, y, w, h, vx, vy, vw, vh]^T

    // 状态协方差 P (8x8)
    cv::Mat P_;

    // 状态转移矩阵 F (8x8) — 恒定速度模型
    cv::Mat F_;

    // 观测矩阵 H (4x8)
    cv::Mat H_;

    // 过程噪声协方差 Q (8x8)
    cv::Mat Q_;

    // 观测噪声协方差 R (4x4)
    cv::Mat R_;

    /// 计算雅可比矩阵 (非线性状态转移时使用)
    cv::Mat computeJacobian() const;

    /// 构建默认矩阵
    void buildMatrices();
};
