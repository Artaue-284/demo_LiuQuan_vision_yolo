/**
 * @file    main.cpp
 * @brief   RoboMaster 装甲板检测 & 跟踪 & 位姿估计 主程序
 *
 * 完整管线:
 *   图像/视频 → YOLO ONNX 检测 → EKF 多目标跟踪 → PnP 位姿估计 → 可视化 → 保存结果
 *
 * 用法:
 *   rm_armor_detect <model.onnx> <image_or_video>
 *   rm_armor_detect <model.onnx> <image_or_video> <output_dir>
 *
 * 示例:
 *   rm_armor_detect ../models/best.onnx ../assets/test.jpg
 *   rm_armor_detect ../models/best.onnx ../assets/video.mp4 ../results/
 */

#include "Detector.h"
#include "Tracker.h"
#include "PnPSolver.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
//  常量
// ============================================================
constexpr int    INPUT_SIZE       = 640;       // 模型输入尺寸
constexpr float  CONF_THRESHOLD   = 0.5f;      // 置信度阈值
constexpr float  NMS_THRESHOLD    = 0.4f;      // NMS IoU 阈值
constexpr int    MAX_TRACKER_MISS = 30;        // EKF 最大丢失帧数
constexpr float  TRACKER_IOU_THR  = 0.3f;      // 跟踪器匹配 IoU 阈值

// 6 分类名称
const std::vector<std::string> CLASS_NAMES = {
    "red3", "red1", "redsb",
    "blue3", "blue1", "bluesb"
};

// BGR 颜色表 (每种类别)
const std::vector<cv::Scalar> CLASS_COLORS = {
    cv::Scalar(0,   0,   255),   // red3  — 红
    cv::Scalar(0,   100, 255),   // red1  — 橙红
    cv::Scalar(0,   165, 255),   // redsb — 橙
    cv::Scalar(255, 0,   0),     // blue3 — 蓝
    cv::Scalar(255, 100, 0),     // blue1 — 蓝绿
    cv::Scalar(255, 165, 0),     // bluesb— 天蓝
};

// 默认相机内参 (近似值, 需根据实际相机标定)
// 使用普通的 1080p 相机, HFOV ≈ 70°
cv::Mat defaultCameraMatrix()
{
    // 假设图像宽度 1920, HFOV 70°, 等效焦距 ≈ 1371
    float fx = 1371.f, fy = 1371.f;   // 焦距 (像素)
    float cx = 960.f,  cy = 540.f;    // 主点 (像素)
    return (cv::Mat_<double>(3, 3) <<
            fx,  0.0, cx,
            0.0, fy,  cy,
            0.0, 0.0, 1.0);
}

// ============================================================
//  工具函数
// ============================================================

/// 在图像上绘制检测/跟踪结果
void drawResults(cv::Mat& image,
                 const std::vector<TrackResult>& tracks,
                 const std::vector<Detection>&   detections,
                 double fps,
                 const PnPSolver& pnp)
{
    // --- 绘制检测框 (细线, 无 ID) ---
    for (const auto& det : detections) {
        cv::Scalar color = (det.class_id >= 0 && det.class_id < static_cast<int>(CLASS_COLORS.size()))
                           ? CLASS_COLORS[det.class_id]
                           : cv::Scalar(0, 255, 0);

        cv::Rect box(static_cast<int>(det.bbox.x),
                     static_cast<int>(det.bbox.y),
                     static_cast<int>(det.bbox.width),
                     static_cast<int>(det.bbox.height));
        cv::rectangle(image, box, color, 1);

        std::string label = cv::format("%s %.2f",
                                        det.class_name.c_str(), det.confidence);
        cv::putText(image, label,
                    cv::Point(box.x, box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }

    // --- 绘制跟踪结果 (粗线, 带 ID + 距离) ---
    for (const auto& tr : tracks) {
        cv::Scalar color = (tr.class_id >= 0 && tr.class_id < static_cast<int>(CLASS_COLORS.size()))
                           ? CLASS_COLORS[tr.class_id]
                           : cv::Scalar(0, 255, 0);

        float cx = tr.state[0], cy = tr.state[1];
        float w  = tr.state[2], h  = tr.state[3];
        cv::Rect2f bbox(cx - w/2.f, cy - h/2.f, w, h);
        cv::Rect  box(static_cast<int>(bbox.x), static_cast<int>(bbox.y),
                      static_cast<int>(bbox.width), static_cast<int>(bbox.height));

        // EKF 预测框 (粗实线, 与检测框区分)
        cv::rectangle(image, box, color, 2);

        // PnP 位姿估计
        ArmorPlateSize size = PnPSolver::sizeForClass(tr.class_id);
        Pose6DOF pose = pnp.solveFromBbox(bbox, size);

        std::string label;
        if (pose.valid) {
            label = cv::format("ID:%d %s %.0fmm",
                               tr.id, tr.class_name.c_str(), pose.distance);
        } else {
            label = cv::format("ID:%d %s",
                               tr.id, tr.class_name.c_str());
        }

        // 背景框
        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                           0.5, 2, &baseline);
        cv::rectangle(image,
                      cv::Point(box.x, box.y - text_sz.height - 8),
                      cv::Point(box.x + text_sz.width, box.y),
                      color, cv::FILLED);
        cv::putText(image, label,
                    cv::Point(box.x, box.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 2);
    }

    // --- FPS 显示 (左上角) ---
    std::string fps_text = cv::format("FPS: %.1f", fps);
    cv::putText(image, fps_text,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);

    // --- 跟踪器数量 ---
    std::string count_text = cv::format("Trackers: %zu", tracks.size());
    cv::putText(image, count_text,
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);
}

/// 处理单帧图像
std::tuple<std::vector<Detection>, std::vector<TrackResult>, double>
processFrame(const cv::Mat& frame,
             Detector& detector,
             Tracker&  tracker)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) YOLO 检测
    auto detections = detector.detect(frame);

    auto t1 = std::chrono::high_resolution_clock::now();

    // 2) EKF 多目标跟踪
    auto tracks = tracker.update(detections);

    auto t2 = std::chrono::high_resolution_clock::now();

    // 计算总耗时 (ms) 和 FPS
    double elapsed = std::chrono::duration<double, std::milli>(t2 - t0).count();
    double fps = (elapsed > 0.0) ? 1000.0 / elapsed : 0.0;

    return {detections, tracks, fps};
}

// ============================================================
//  processImage — 处理单张图像
// ============================================================
int processImage(const std::string& model_path,
                 const std::string& image_path,
                 const std::string& output_dir)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "错误: 无法读取图像 " << image_path << std::endl;
        return 1;
    }

    std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;

    // 初始化组件
    Detector  detector(model_path, CLASS_NAMES, CONF_THRESHOLD, NMS_THRESHOLD, INPUT_SIZE);
    Tracker   tracker(MAX_TRACKER_MISS, TRACKER_IOU_THR);
    PnPSolver pnp(defaultCameraMatrix());

    // 处理
    auto [detections, tracks, fps] = processFrame(image, detector, tracker);

    std::cout << "检测到 " << detections.size() << " 个目标, "
              << "跟踪 " << tracks.size() << " 个目标" << std::endl;
    std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;

    // 打印每个跟踪目标的详细信息
    for (const auto& tr : tracks) {
        ArmorPlateSize size = PnPSolver::sizeForClass(tr.class_id);
        cv::Rect2f bbox(tr.state[0] - tr.state[2]/2.f,
                        tr.state[1] - tr.state[3]/2.f,
                        tr.state[2], tr.state[3]);
        Pose6DOF pose = pnp.solveFromBbox(bbox, size);
        std::cout << "  ID:" << tr.id
                  << " 类别:" << tr.class_name
                  << " 置信度:" << std::fixed << std::setprecision(3) << tr.confidence
                  << " 位置:(" << std::fixed << std::setprecision(1)
                  << tr.state[0] << "," << tr.state[1] << ")"
                  << " 尺寸:(" << tr.state[2] << "x" << tr.state[3] << ")";
        if (pose.valid) {
            std::cout << " 距离:" << std::fixed << std::setprecision(1)
                      << pose.distance << "mm";
        }
        std::cout << std::endl;
    }

    // 可视化
    cv::Mat vis = image.clone();
    drawResults(vis, tracks, detections, fps, pnp);

    // 保存结果
    fs::create_directories(output_dir);
    std::string out_name = fs::path(image_path).stem().string() + "_result.jpg";
    std::string out_path = (fs::path(output_dir) / out_name).string();
    cv::imwrite(out_path, vis);
    std::cout << "结果已保存: " << out_path << std::endl;

    // 显示 (可选)
    cv::namedWindow("RoboMaster Armor Detect", cv::WINDOW_NORMAL);
    cv::imshow("RoboMaster Armor Detect", vis);
    std::cout << "按任意键退出..." << std::endl;
    cv::waitKey(0);

    return 0;
}

// ============================================================
//  processVideo — 处理视频
// ============================================================
int processVideo(const std::string& model_path,
                 const std::string& video_path,
                 const std::string& output_dir)
{
    cv::VideoCapture cap;
    if (video_path == "0" || video_path == "camera") {
        cap.open(0);  // 摄像头
    } else {
        cap.open(video_path);
    }

    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频源 " << video_path << std::endl;
        return 1;
    }

    double cap_fps = cap.get(cv::CAP_PROP_FPS);
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "视频: " << width << "x" << height << " @ " << cap_fps << "fps" << std::endl;

    // 初始化组件
    Detector  detector(model_path, CLASS_NAMES, CONF_THRESHOLD, NMS_THRESHOLD, INPUT_SIZE);
    Tracker   tracker(MAX_TRACKER_MISS, TRACKER_IOU_THR);

    // 根据视频分辨率调整相机矩阵
    cv::Mat cam_mat = defaultCameraMatrix();
    double scale_x = width / 1920.0;
    double scale_y = height / 1080.0;
    cam_mat.at<double>(0, 0) *= scale_x;
    cam_mat.at<double>(1, 1) *= scale_y;
    cam_mat.at<double>(0, 2) *= scale_x;
    cam_mat.at<double>(1, 2) *= scale_y;
    PnPSolver pnp(cam_mat);

    // 输出视频
    fs::create_directories(output_dir);
    std::string out_video = (fs::path(output_dir) / "output.avi").string();
    cv::VideoWriter writer(out_video,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           cap_fps > 0 ? cap_fps : 30.0,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "警告: 无法创建输出视频, 将只显示不保存" << std::endl;
    }

    // 累计 FPS 统计
    double total_fps = 0.0;
    int    frame_count = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        auto [detections, tracks, fps] = processFrame(frame, detector, tracker);

        total_fps += fps;
        frame_count++;

        // 可视化
        drawResults(frame, tracks, detections, fps, pnp);

        // 写入输出视频
        if (writer.isOpened()) {
            writer.write(frame);
        }

        // 显示
        cv::imshow("RoboMaster Armor Detect", frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {  // ESC 或 q 退出
            std::cout << "用户中止" << std::endl;
            break;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();

    double avg_fps = frame_count > 0 ? total_fps / frame_count : 0.0;
    std::cout << "\n===== 处理完成 =====" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "平均 FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;
    if (!out_video.empty()) {
        std::cout << "输出视频: " << out_video << std::endl;
    }

    return 0;
}

// ============================================================
//  printUsage
// ============================================================
void printUsage(const char* prog)
{
    std::cout << "RoboMaster 装甲板检测 & 跟踪 & 位姿估计\n"
              << "================================================\n\n"
              << "用法:\n"
              << "  " << prog << " <model.onnx> <image>\n"
              << "  " << prog << " <model.onnx> <image> <output_dir>\n"
              << "  " << prog << " <model.onnx> <video> [output_dir]\n"
              << "  " << prog << " <model.onnx> camera [output_dir]\n\n"
              << "参数:\n"
              << "  model.onnx   ONNX 模型文件路径\n"
              << "  image/video  输入图像或视频文件 (或 'camera' 使用摄像头)\n"
              << "  output_dir   输出目录 (默认: ../results)\n\n"
              << "示例:\n"
              << "  " << prog << " ../models/best.onnx ../assets/test.jpg\n"
              << "  " << prog << " ../models/best.onnx ../assets/video.mp4 ../results/\n"
              << std::endl;
}

// ============================================================
//  main
// ============================================================
int main(int argc, char* argv[])
{
    if (argc < 3) {
        printUsage(argv[0]);
        return 0;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string output_dir = (argc >= 4) ? argv[3] : "../results";

    // 检查模型文件
    if (!fs::exists(model_path)) {
        std::cerr << "错误: 模型文件不存在: " << model_path << std::endl;
        return 1;
    }

    std::cout << "RoboMaster 装甲板检测系统\n"
              << "模型: " << model_path << "\n"
              << "输入: " << input_path << "\n"
              << "输出: " << output_dir << "\n"
              << "类别: ";
    for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        std::cout << CLASS_NAMES[i] << (i < CLASS_NAMES.size()-1 ? ", " : "\n");
    }
    std::cout << std::endl;

    // 判断输入类型
    std::string ext = fs::path(input_path).extension().string();
    bool is_image = (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
    bool is_video = (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv");
    bool is_camera = (input_path == "0" || input_path == "camera");

    if (is_image) {
        return processImage(model_path, input_path, output_dir);
    } else if (is_video || is_camera) {
        return processVideo(model_path, input_path, output_dir);
    } else {
        // 尝试作为图像打开
        cv::Mat test = cv::imread(input_path);
        if (!test.empty()) {
            return processImage(model_path, input_path, output_dir);
        }
        std::cerr << "错误: 无法识别的输入格式: " << input_path << std::endl;
        return 1;
    }
}
