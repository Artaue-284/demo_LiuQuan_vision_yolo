/**
 * @file    Tracker.h
 * @brief   多目标跟踪器 — 基于 EKF + IoU 贪心匹配
 *
 * 工作原理:
 *   1. 对所有现有 trackers 执行 predict()
 *   2. 计算 tracker 预测位置与 detection 的 IoU 矩阵
 *   3. 使用贪心算法进行数据关联 (距离最近者优先)
 *   4. 匹配成功的 tracker 执行 update()
 *   5. 未匹配的 detection 创建新 tracker
 *   6. 连续丢失超过 max_miss_ 帧的 tracker 被删除
 */
#pragma once

#include "KalmanFilter.h"
#include "Detector.h"

#include <vector>
#include <memory>

// ============================================================
//  跟踪结果
// ============================================================
struct TrackResult {
    int         id;             // 跟踪 ID
    cv::Vec4f   state;          // [x, y, w, h] (滤波后)
    cv::Vec4f   velocity;       // [vx, vy, vw, vh]
    int         class_id;       // 类别编号
    float       confidence;     // 最新检测置信度
    std::string class_name;     // 类别名称
    int         age;            // 跟踪持续帧数
};

// ============================================================
//  多目标跟踪器
// ============================================================
class Tracker {
public:
    /**
     * @param max_miss  最大允许丢失帧数 (超过则删除)
     * @param iou_threshold  匹配 IoU 阈值
     */
    explicit Tracker(int max_miss = 30, float iou_threshold = 0.3f);

    /**
     * @brief  用新的检测结果更新所有跟踪器
     * @param  detections  当前帧的检测结果
     * @return 活跃的跟踪结果列表
     */
    std::vector<TrackResult> update(const std::vector<Detection>& detections);

    /// 获取当前活跃跟踪器数量
    size_t activeCount() const;

private:
    std::vector<std::shared_ptr<KalmanFilter>> trackers_;  // 活跃跟踪器
    std::vector<int>            class_ids_;                 // 对应类别
    std::vector<std::string>    class_names_;               // 类别名称
    std::vector<float>          confidences_;               // 最新置信度
    std::vector<int>            ages_;                      // 持续帧数

    int    max_miss_;        // 最大丢失帧数
    float  iou_threshold_;   // 匹配 IoU 阈值

    /**
     * @brief  计算两个边界框的 IoU
     */
    static float computeIoU(const cv::Vec4f& a, const cv::Vec4f& b);

    /**
     * @brief  贪心数据关联
     * @param  trackers   跟踪器预测位置列表
     * @param  detections 检测结果列表
     * @return 匹配对 (tracker_idx, detection_idx), 未匹配的另作处理
     */
    std::vector<std::pair<int, int>> associate(
        const std::vector<cv::Vec4f>&  tracker_boxes,
        const std::vector<Detection>&  detections);
};
