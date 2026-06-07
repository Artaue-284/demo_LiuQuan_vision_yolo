/**
 * @file    PnPSolver.cpp
 * @brief   PnP 位姿估计器实现
 */

#include "PnPSolver.h"

#include <cmath>

// ============================================================
//  构造
// ============================================================
PnPSolver::PnPSolver(const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs)
{
    camera_matrix.copyTo(K_);
    dist_coeffs.copyTo(dist_);
}

// ============================================================
//  solve — 标准 PnP 求解
// ============================================================
Pose6DOF PnPSolver::solve(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points) const
{
    Pose6DOF pose;

    if (image_points.size() < 4 || object_points.size() < 4) {
        return pose;  // 至少需要 4 个点
    }

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(
        object_points, image_points,
        K_, dist_,
        rvec, tvec,
        false,                     // useExtrinsicGuess
        cv::SOLVEPNP_IPPE_SQUARE   // 适合平面矩形目标
    );

    if (!success) {
        // 回退到迭代法
        success = cv::solvePnP(
            object_points, image_points,
            K_, dist_,
            rvec, tvec,
            false,
            cv::SOLVEPNP_ITERATIVE);
    }

    if (success) {
        pose.rvec = cv::Vec3d(rvec.at<double>(0),
                              rvec.at<double>(1),
                              rvec.at<double>(2));
        pose.tvec = cv::Vec3d(tvec.at<double>(0),
                              tvec.at<double>(1),
                              tvec.at<double>(2));
        pose.distance = static_cast<float>(
            cv::norm(cv::Vec3d(pose.tvec)));
        pose.valid = true;
    }

    return pose;
}

// ============================================================
//  solveFromBbox — 从检测框快速估计位姿
// ============================================================
Pose6DOF PnPSolver::solveFromBbox(
    const cv::Rect2f& bbox,
    const ArmorPlateSize& plate_size) const
{
    // 2D 点: 检测框的四个角点 (左上, 右上, 右下, 左下)
    std::vector<cv::Point2f> img_pts = {
        cv::Point2f(bbox.x,                bbox.y),                 // TL
        cv::Point2f(bbox.x + bbox.width,   bbox.y),                 // TR
        cv::Point2f(bbox.x + bbox.width,   bbox.y + bbox.height),   // BR
        cv::Point2f(bbox.x,                bbox.y + bbox.height),   // BL
    };

    // 3D 点: 装甲板在 Z=0 平面, 中心在原点
    float hw = plate_size.width  / 2.f;   // 半宽
    float hh = plate_size.height / 2.f;   // 半高

    std::vector<cv::Point3f> obj_pts = {
        cv::Point3f(-hw, -hh, 0.f),   // TL
        cv::Point3f( hw, -hh, 0.f),   // TR
        cv::Point3f( hw,  hh, 0.f),   // BR
        cv::Point3f(-hw,  hh, 0.f),   // BL
    };

    return solve(img_pts, obj_pts);
}

// ============================================================
//  sizeForClass — 根据类别 ID 返回装甲板尺寸
// ============================================================
ArmorPlateSize PnPSolver::sizeForClass(int class_id)
{
    switch (class_id) {
    case 0:  // red3  — 红方步兵 (3号装甲板)
    case 3:  // blue3 — 蓝方步兵
        return ArmorSizes::SOLDIER_3;

    case 1:  // red1  — 红方英雄 (1号装甲板)
    case 4:  // blue1 — 蓝方英雄
        return ArmorSizes::HERO_1;

    case 2:  // redsb  — 红方哨兵
    case 5:  // bluesb — 蓝方哨兵
        return ArmorSizes::SENTRY;

    default:
        return ArmorSizes::DEFAULT;
    }
}
