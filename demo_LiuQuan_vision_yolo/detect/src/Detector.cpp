/**
 * @file    Detector.cpp
 * @brief   YOLO ONNX 检测器实现
 */

#include "Detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <map>

// ============================================================
//  Detector 构造
// ============================================================
Detector::Detector(const std::string&              model_path,
                   const std::vector<std::string>& class_names,
                   float conf_threshold,
                   float nms_threshold,
                   int   input_size)
    : class_names_(class_names)
    , conf_threshold_(conf_threshold)
    , nms_threshold_(nms_threshold)
    , input_size_(input_size)
{
    // 使用 OpenCV DNN 加载 ONNX 模型
    net_ = cv::dnn::readNetFromONNX(model_path);

    // 设置后端和目标
    // CUDA 后端 (如果 OpenCV 以 CUDA 编译)
    // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // CPU 后端 (默认, 兼容性最好)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 尝试使用 OpenVINO 后端加速
    // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// ============================================================
//  detect — 主检测流程
// ============================================================
std::vector<Detection> Detector::detect(const cv::Mat& image)
{
    if (image.empty()) return {};

    cv::Size original_size(image.cols, image.rows);

    // 1) 预处理
    cv::Mat blob = preprocess(image);

    // 2) 推理
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.empty()) return {};

    // 3) 后处理
    return postprocess(outputs, original_size);
}

// ============================================================
//  preprocess — 图像 → blob
// ============================================================
cv::Mat Detector::preprocess(const cv::Mat& image)
{
    // 1) 保持宽高比的 letterbox resize
    cv::Mat resized;
    float scale = std::min(
        static_cast<float>(input_size_) / image.cols,
        static_cast<float>(input_size_) / image.rows);

    int new_w = static_cast<int>(image.cols * scale);
    int new_h = static_cast<int>(image.rows * scale);
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // 2) 填充到 input_size x input_size
    int dw = input_size_ - new_w;
    int dh = input_size_ - new_h;
    int top  = dh / 2;
    int bot  = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::copyMakeBorder(resized, resized, top, bot, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 3) 转为 blob: [1, 3, 640, 640], BGR→RGB, 归一化到 [0,1]
    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(input_size_, input_size_),
        cv::Scalar(), true, false);

    return blob;
}

// ============================================================
//  postprocess — 解码 + NMS
// ============================================================
std::vector<Detection> Detector::postprocess(
    const std::vector<cv::Mat>& outputs,
    const cv::Size&             original_size)
{
    // YOLO11 ONNX 通常只有 1 个输出
    const cv::Mat& output = outputs[0];

    // 解码 YOLO 输出
    std::vector<Detection> detections = decodeYoloOutput(
        output,
        original_size.width,
        original_size.height,
        conf_threshold_,
        class_names_);

    // 按类别独立做 NMS
    std::vector<Detection> final_dets = applyNMS(detections, nms_threshold_);

    return final_dets;
}

// ============================================================
//  decodeYoloOutput — 解码 YOLO 输出张量
// ============================================================
std::vector<Detection> decodeYoloOutput(
    const cv::Mat&                   output,
    int img_w, int img_h,
    float conf_threshold,
    const std::vector<std::string>&   class_names)
{
    std::vector<Detection> detections;

    // output shape: [1, 4 + num_classes, num_proposals]
    //   对 6 分类模型: [1, 10, 8400]
    const int dims = output.dims;
    if (dims != 3) {
        // 可能是 [1, 8400, 10] (已转置), 尝试另一种解析
        // 返回空, 外部可尝试 reshape 后再调用
        return detections;
    }

    const int num_classes = static_cast<int>(class_names.size());
    const int stride      = 4 + num_classes;   // 每列的 stride 大小
    const int num_proposals = output.size[2];   // 候选框数量

    // 检查输出格式是否匹配
    if (output.size[1] != stride) {
        // 尝试处理转置格式 [1, num_proposals, stride]
        if (output.size[2] == stride) {
            // 手动转置最后两维: [1, P, S] → [1, S, P]
            std::vector<int> new_shape = {1, output.size[2], output.size[1]};
            cv::Mat transposed = output.reshape(1, new_shape);
            // 此时 reshape 不会真的重排数据, 所以我们直接按列读取
            // 更稳健的做法是遍历每个 proposal
            std::vector<Detection> detections;
            int n_prop = output.size[1];
            int n_cls = static_cast<int>(class_names.size());
            for (int i = 0; i < n_prop; ++i) {
                float cx = output.at<float>(0, i, 0);
                float cy = output.at<float>(0, i, 1);
                float w  = output.at<float>(0, i, 2);
                float h  = output.at<float>(0, i, 3);
                float obj = output.at<float>(0, i, 4);

                float max_score = 0.f;
                int   best_cls  = -1;
                for (int c = 0; c < n_cls; ++c) {
                    float cls_s = output.at<float>(0, i, 5 + c);
                    float score = obj * cls_s;
                    if (score > max_score) { max_score = score; best_cls = c; }
                }
                if (max_score < conf_threshold) continue;

                Detection det;
                det.bbox.x      = (cx - w / 2.f) * img_w;
                det.bbox.y      = (cy - h / 2.f) * img_h;
                det.bbox.width  = w * img_w;
                det.bbox.height = h * img_h;
                det.class_id    = best_cls;
                det.confidence  = max_score;
                det.class_name  = (best_cls >= 0 && best_cls < n_cls)
                                    ? class_names[best_cls] : "unknown";
                det.bbox.x      = std::max(0.f, det.bbox.x);
                det.bbox.y      = std::max(0.f, det.bbox.y);
                det.bbox.width  = std::min(det.bbox.width,  img_w - det.bbox.x);
                det.bbox.height = std::min(det.bbox.height, img_h - det.bbox.y);
                if (det.bbox.width > 0 && det.bbox.height > 0)
                    detections.push_back(det);
            }
            return detections;
        }
        // 无法识别的格式
        return detections;
    }

    // 遍历所有候选框
    for (int i = 0; i < num_proposals; ++i) {
        // 获取 objectness + class scores 的最大值
        float max_score = 0.f;
        int   best_cls  = -1;

        // 先用原始 objectness (索引 4)
        float obj_conf = output.at<float>(0, 4, i);

        for (int c = 0; c < num_classes; ++c) {
            float cls_score = output.at<float>(0, 5 + c, i);
            float score = obj_conf * cls_score;  // 联合置信度

            if (score > max_score) {
                max_score = score;
                best_cls  = c;
            }
        }

        if (max_score < conf_threshold) continue;

        // 提取坐标 (已归一化到 [0, 1])
        float cx = output.at<float>(0, 0, i);
        float cy = output.at<float>(0, 1, i);
        float w  = output.at<float>(0, 2, i);
        float h  = output.at<float>(0, 3, i);

        // 转为像素坐标
        Detection det;
        det.bbox.x      = (cx - w / 2.f) * img_w;
        det.bbox.y      = (cy - h / 2.f) * img_h;
        det.bbox.width  = w * img_w;
        det.bbox.height = h * img_h;
        det.class_id    = best_cls;
        det.confidence  = max_score;
        det.class_name  = (best_cls >= 0 && best_cls < static_cast<int>(class_names.size()))
                            ? class_names[best_cls] : "unknown";

        // 边界裁剪
        det.bbox.x      = std::max(0.f, det.bbox.x);
        det.bbox.y      = std::max(0.f, det.bbox.y);
        det.bbox.width  = std::min(det.bbox.width,  img_w - det.bbox.x);
        det.bbox.height = std::min(det.bbox.height, img_h - det.bbox.y);

        if (det.bbox.width > 0 && det.bbox.height > 0) {
            detections.push_back(det);
        }
    }

    return detections;
}

// ============================================================
//  applyNMS — 按类别分别执行非极大值抑制
// ============================================================
std::vector<Detection> applyNMS(
    const std::vector<Detection>& detections,
    float iou_threshold)
{
    if (detections.empty()) return {};

    // 按类别分组
    std::map<int, std::vector<Detection>> class_groups;
    for (const auto& d : detections) {
        class_groups[d.class_id].push_back(d);
    }

    std::vector<Detection> result;

    for (auto& [cls_id, dets] : class_groups) {
        // 转换为 OpenCV NMS 需要的格式
        std::vector<cv::Rect> boxes;
        std::vector<float>   scores;
        for (const auto& d : dets) {
            boxes.push_back(cv::Rect(
                static_cast<int>(d.bbox.x),
                static_cast<int>(d.bbox.y),
                static_cast<int>(d.bbox.width),
                static_cast<int>(d.bbox.height)));
            scores.push_back(d.confidence);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, 0.f, iou_threshold, keep);

        for (int idx : keep) {
            result.push_back(dets[idx]);
        }
    }

    return result;
}
