/**
 * @file    Tracker.cpp
 * @brief   多目标跟踪器实现
 */

#include "Tracker.h"

#include <algorithm>
#include <limits>
#include <set>

// ============================================================
//  构造
// ============================================================
Tracker::Tracker(int max_miss, float iou_threshold)
    : max_miss_(max_miss)
    , iou_threshold_(iou_threshold)
{
}

// ============================================================
//  update — 主更新循环
// ============================================================
std::vector<TrackResult> Tracker::update(const std::vector<Detection>& detections)
{
    // ----- 1) 对所有跟踪器执行预测 -----
    std::vector<cv::Vec4f> pred_boxes;
    pred_boxes.reserve(trackers_.size());
    for (auto& t : trackers_) {
        pred_boxes.push_back(t->predict());
    }

    // ----- 2) 数据关联 -----
    auto matches = associate(pred_boxes, detections);

    std::set<int> matched_trackers;
    std::set<int> matched_detections;

    for (auto& [t_idx, d_idx] : matches) {
        matched_trackers.insert(t_idx);
        matched_detections.insert(d_idx);

        const auto& det = detections[d_idx];
        trackers_[t_idx]->update(
            det.bbox.x + det.bbox.width  / 2.f,
            det.bbox.y + det.bbox.height / 2.f,
            det.bbox.width,
            det.bbox.height);
        class_ids_[t_idx]   = det.class_id;
        class_names_[t_idx] = det.class_name;
        confidences_[t_idx] = det.confidence;
        ages_[t_idx]++;
    }

    // ----- 3) 未匹配的跟踪器: 增加丢失计数 -----
    for (size_t i = 0; i < trackers_.size(); ++i) {
        if (matched_trackers.count(static_cast<int>(i)) == 0) {
            trackers_[i]->incrementMiss();
        }
    }

    // ----- 4) 未匹配的检测: 创建新跟踪器 -----
    for (size_t i = 0; i < detections.size(); ++i) {
        if (matched_detections.count(static_cast<int>(i)) == 0) {
            const auto& det = detections[i];
            auto kf = std::make_shared<KalmanFilter>();
            kf->init(
                det.bbox.x + det.bbox.width  / 2.f,
                det.bbox.y + det.bbox.height / 2.f,
                det.bbox.width,
                det.bbox.height);
            trackers_.push_back(kf);
            class_ids_.push_back(det.class_id);
            class_names_.push_back(det.class_name);
            confidences_.push_back(det.confidence);
            ages_.push_back(1);
        }
    }

    // ----- 5) 清理丢失过多的跟踪器 -----
    std::vector<size_t> to_remove;
    for (size_t i = 0; i < trackers_.size(); ++i) {
        if (trackers_[i]->missCount() > max_miss_) {
            to_remove.push_back(i);
        }
    }
    // 从后往前删除, 避免索引偏移
    std::sort(to_remove.begin(), to_remove.end(), std::greater<size_t>());
    for (size_t idx : to_remove) {
        trackers_.erase(trackers_.begin() + idx);
        class_ids_.erase(class_ids_.begin() + idx);
        class_names_.erase(class_names_.begin() + idx);
        confidences_.erase(confidences_.begin() + idx);
        ages_.erase(ages_.begin() + idx);
    }

    // ----- 6) 返回活跃结果 -----
    std::vector<TrackResult> results;
    results.reserve(trackers_.size());
    for (size_t i = 0; i < trackers_.size(); ++i) {
        TrackResult tr;
        tr.id         = trackers_[i]->id();
        tr.state      = trackers_[i]->state();
        tr.velocity   = trackers_[i]->velocity();
        tr.class_id   = class_ids_[i];
        tr.confidence = confidences_[i];
        tr.class_name = class_names_[i];
        tr.age        = ages_[i];
        results.push_back(tr);
    }
    return results;
}

// ============================================================
//  computeIoU
// ============================================================
float Tracker::computeIoU(const cv::Vec4f& a, const cv::Vec4f& b)
{
    // a, b 格式: [cx, cy, w, h]
    float ax1 = a[0] - a[2] / 2.f;
    float ay1 = a[1] - a[3] / 2.f;
    float ax2 = a[0] + a[2] / 2.f;
    float ay2 = a[1] + a[3] / 2.f;

    float bx1 = b[0] - b[2] / 2.f;
    float by1 = b[1] - b[3] / 2.f;
    float bx2 = b[0] + b[2] / 2.f;
    float by2 = b[1] + b[3] / 2.f;

    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);

    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;

    float area_a = a[2] * a[3];
    float area_b = b[2] * b[3];
    float uni = area_a + area_b - inter;

    return (uni > 1e-6f) ? inter / uni : 0.f;
}

// ============================================================
//  associate — 贪心 IoU 匹配
// ============================================================
std::vector<std::pair<int, int>> Tracker::associate(
    const std::vector<cv::Vec4f>&  tracker_boxes,
    const std::vector<Detection>&  detections)
{
    std::vector<std::pair<int, int>> matches;

    if (tracker_boxes.empty() || detections.empty())
        return matches;

    // 构建 IoU 矩阵 + 候选匹配列表
    struct Candidate {
        int   t_idx;
        int   d_idx;
        float iou;
    };
    std::vector<Candidate> candidates;

    for (size_t t = 0; t < tracker_boxes.size(); ++t) {
        cv::Vec4f det_box;
        for (size_t d = 0; d < detections.size(); ++d) {
            const auto& b = detections[d].bbox;
            det_box = cv::Vec4f(b.x + b.width / 2.f, b.y + b.height / 2.f,
                                b.width, b.height);
            float iou = computeIoU(tracker_boxes[t], det_box);
            if (iou >= iou_threshold_) {
                candidates.push_back({static_cast<int>(t), static_cast<int>(d), iou});
            }
        }
    }

    // 按 IoU 降序排序 (贪心: 最佳匹配优先)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.iou > b.iou; });

    std::set<int> used_t, used_d;
    for (const auto& cand : candidates) {
        if (used_t.count(cand.t_idx) == 0 && used_d.count(cand.d_idx) == 0) {
            matches.emplace_back(cand.t_idx, cand.d_idx);
            used_t.insert(cand.t_idx);
            used_d.insert(cand.d_idx);
        }
    }

    return matches;
}

// ============================================================
//  activeCount
// ============================================================
size_t Tracker::activeCount() const
{
    return trackers_.size();
}
