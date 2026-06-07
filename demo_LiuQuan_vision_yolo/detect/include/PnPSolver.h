/**
 * @file    PnPSolver.h
 * @brief   PnP (Perspective-n-Point) 位姿估计器
 *
 * 使用 solvePnP 从 2D 图像点与 3D 模型点的对应关系估计 6-DOF 位姿。
 *
 * 装甲板模型点 (根据 RoboMaster 官方规则手册):
 *   - 步兵装甲板: 约 230mm × 125mm (含灯条外框)
 *   - 英雄装甲板: 约 340mm × 125mm
 *   - 哨兵装甲板: 约 200mm × 100mm
 *
 * 2D 点来源: YOLO 检测框的四个角点
 * 3D 点: 装甲板在世界坐标系中的对应角点 (Z=0 平面)
 */
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>
#include <array>

// ============================================================
//  装甲板物理尺寸 (单位: 毫米)
// ============================================================
struct ArmorPlateSize {
    float width;   // 宽度 (mm)
    float height;  // 高度 (mm)
};

// 预定义尺寸 (根据 RoboMaster 规则)
namespace ArmorSizes {
    constexpr ArmorPlateSize SOLDIER_3  { 230.f, 125.f };   // 步兵 (3号)
    constexpr ArmorPlateSize HERO_1     { 340.f, 125.f };   // 英雄 (1号)
    constexpr ArmorPlateSize SENTRY     { 200.f, 100.f };   // 哨兵
    constexpr ArmorPlateSize DEFAULT    { 230.f, 125.f };   // 默认使用步兵尺寸
}

// ============================================================
//  PnP 结果
// ============================================================
struct Pose6DOF {
    cv::Vec3d   rvec;        // 旋转向量 (Rodrigues)
    cv::Vec3d   tvec;        // 平移向量 (mm, 相机坐标系)
    float       distance;    // 相机到目标的欧氏距离 (mm)
    bool        valid;       // 估计是否有效

    Pose6DOF() : rvec(0, 0, 0), tvec(0, 0, 0), distance(0.f), valid(false) {}
};

// ============================================================
//  PnP 求解器
// ============================================================
class PnPSolver {
public:
    /**
     * @param camera_matrix   相机内参矩阵 (3x3)
     * @param dist_coeffs     畸变系数 (默认无畸变)
     */
    explicit PnPSolver(const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs = cv::Mat::zeros(5, 1, CV_64F));

    /**
     * @brief  估计装甲板的 6-DOF 位姿
     * @param  image_points   2D 图像点 (检测框的四个角点, 像素坐标)
     * @param  object_points  3D 模型点 (装甲板角点, mm)
     * @return Pose6DOF 结构
     */
    Pose6DOF solve(const std::vector<cv::Point2f>& image_points,
                   const std::vector<cv::Point3f>& object_points) const;

    /**
     * @brief  从 YOLO 检测框快速估计位姿
     *         使用检测框的四个角点作为 2D 点,
     *         使用装甲板尺寸构建 3D 模型点
     * @param  bbox    检测框 (x, y, w, h 像素坐标)
     * @param  plate_size  装甲板物理尺寸
     * @return Pose6DOF 结构
     */
    Pose6DOF solveFromBbox(const cv::Rect2f& bbox,
                           const ArmorPlateSize& plate_size = ArmorSizes::DEFAULT) const;

    /// 根据类别 ID 获取装甲板尺寸
    static ArmorPlateSize sizeForClass(int class_id);

    /// 设置相机内参
    void setCameraMatrix(const cv::Mat& K) { K.copyTo(K_); }

private:
    cv::Mat K_;      // 相机内参 (3x3)
    cv::Mat dist_;   // 畸变系数
};
