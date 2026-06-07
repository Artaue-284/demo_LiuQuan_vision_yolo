/**
 * @file    Detector.h
 * @brief   YOLO ONNX 装甲板检测器 — 使用 OpenCV DNN 或 ONNX Runtime 推理
 *
 * 支持 6 分类检测:
 *   0: red3    (红方步兵)
 *   1: red1    (红方英雄)
 *   2: redsb   (红方哨兵)
 *   3: blue3   (蓝方步兵)
 *   4: blue1   (蓝方英雄)
 *   5: bluesb  (蓝方哨兵)
 *
 * 使用方式:
 *   Detector det("models/best.onnx", class_names, 0.5f, 0.4f, 640);
 *   auto results = det.detect(image);
 */
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

// ============================================================
//  检测结果结构体
// ============================================================
struct Detection {
    int         class_id   = -1;          // 类别编号 (0-5)
    float       confidence = 0.f;         // 置信度 [0, 1]
    cv::Rect2f  bbox;                     // 边界框 (x, y, w, h) 像素坐标
    std::string class_name;               // 类别名称
};

// ============================================================
//  YOLO 检测器
// ============================================================
class Detector {
public:
    /**
     * @param model_path    ONNX 模型文件路径
     * @param class_names   类别名称列表 (6 个)
     * @param conf_threshold  置信度阈值
     * @param nms_threshold   NMS IoU 阈值
     * @param input_size     模型输入尺寸 (默认 640)
     */
    Detector(const std::string&              model_path,
             const std::vector<std::string>& class_names,
             float conf_threshold = 0.5f,
             float nms_threshold  = 0.4f,
             int   input_size     = 640);

    /**
     * @brief  对单帧图像执行检测
     * @param  image  输入图像 (BGR)
     * @return 检测结果列表
     */
    std::vector<Detection> detect(const cv::Mat& image);

    /// 设置置信度阈值
    void setConfThreshold(float t) { conf_threshold_ = t; }
    /// 设置 NMS 阈值
    void setNmsThreshold(float t)  { nms_threshold_  = t; }

private:
    cv::dnn::Net              net_;              // OpenCV DNN 网络
    std::vector<std::string>  class_names_;      // 类别名称
    float                     conf_threshold_;   // 置信度阈值
    float                     nms_threshold_;    // NMS 阈值
    int                       input_size_;       // 模型输入尺寸

    /**
     * @brief  图像预处理 (resize + normalize + blob)
     */
    cv::Mat preprocess(const cv::Mat& image);

    /**
     * @brief  网络输出后处理 (解码 + NMS)
     * @param  outputs         网络原始输出
     * @param  original_size   原始图像尺寸
     * @return 检测结果列表
     */
    std::vector<Detection> postprocess(
        const std::vector<cv::Mat>& outputs,
        const cv::Size&             original_size);
};

// ============================================================
//  YOLO 模型输出解码 (独立函数, 方便单元测试)
// ============================================================

/**
 * @brief  解析 YOLO11 ONNX 输出张量
 *
 * YOLO11 ONNX 输出 shape: [1, 4 + num_classes, num_proposals]
 *   即 [1, 10, 8400] (6 分类)
 * 每列含义: [cx, cy, w, h, obj_conf, cls0, cls1, cls2, cls3, cls4, cls5]
 * 坐标已归一化到 [0, 1]
 *
 * @param  output          网络输出张量
 * @param  img_w, img_h    原始图像尺寸
 * @param  conf_threshold  置信度阈值
 * @param  class_names     类别名称
 * @return 原始检测框 (NMS 前)
 */
std::vector<Detection> decodeYoloOutput(
    const cv::Mat&                   output,
    int img_w, int img_h,
    float conf_threshold,
    const std::vector<std::string>&   class_names);

/**
 * @brief  NMS (非极大值抑制)
 * @param  detections  原始检测框
 * @param  iou_threshold  IoU 阈值
 * @return 筛选后的检测框
 */
std::vector<Detection> applyNMS(
    const std::vector<Detection>& detections,
    float iou_threshold);
