// Live webcam YOLOv8 detection using a TorchScript-exported .pt model
// with custom CUDA preprocessing + postprocessing.

#include "ts_pipeline.h"
#include "cuda_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

// V4L2: disable autoexposure's dynamic-framerate fallback. Without this, the
// driver silently drops the device FPS (e.g. 30→6) to give autoexposure more
// time per frame in low light — capping cap.read() throughput regardless of
// pixel format or display settings.
static void v4l2_pin_framerate(int cam_id) {
    char path[32];
    snprintf(path, sizeof(path), "/dev/video%d", cam_id);
    int fd = open(path, O_RDWR);
    if (fd < 0) return;
    v4l2_control ctl{};
    ctl.id = 0x009a0903;  // V4L2_CID_EXPOSURE_DYNAMIC_FRAMERATE (UVC)
    ctl.value = 0;
    if (ioctl(fd, VIDIOC_S_CTRL, &ctl) == 0) {
        printf("v4l2: exposure_dynamic_framerate=0 (FPS pinned)\n");
    }
    close(fd);
}

static const char* COCO_NAMES[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static cv::Scalar color_for_class(int cls) {
    static const cv::Scalar palette[8] = {
        {255,  64,  64}, { 64, 255,  64}, { 64,  64, 255}, {255, 255,   0},
        {255,   0, 255}, {  0, 255, 255}, {255, 128,   0}, {128,   0, 255}
    };
    return palette[((unsigned)cls) & 7];
}

static void draw_box(cv::Mat& img, const Detection& d) {
    cv::Scalar col = color_for_class(d.class_id);
    cv::Point p1((int)d.x1, (int)d.y1);
    cv::Point p2((int)d.x2, (int)d.y2);
    cv::rectangle(img, p1, p2, col, 2);
    const char* name = (d.class_id >= 0 && d.class_id < 80)
                       ? COCO_NAMES[d.class_id] : "?";
    char label[128];
    snprintf(label, sizeof(label), "%s %.2f", name, d.score);
    int base = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    cv::Point tl(p1.x, std::max(0, p1.y - ts.height - 4));
    cv::rectangle(img, tl, cv::Point(tl.x + ts.width + 6, tl.y + ts.height + 4),
                  col, cv::FILLED);
    cv::putText(img, label, cv::Point(tl.x + 3, tl.y + ts.height + 1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
}

int main(int argc, char** argv) {
    std::string model = "models/yolov8n.torchscript";
    int cam_id = 0;
    int cam_w = 1280, cam_h = 720;
    float score = 0.25f, iou = 0.45f;
    bool headless = false;
    std::string video_path;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (std::strcmp(argv[i], "--cam")   == 0 && i + 1 < argc) cam_id = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--video") == 0 && i + 1 < argc) video_path = argv[++i];
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) cam_w = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height")== 0 && i + 1 < argc) cam_h = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--score") == 0 && i + 1 < argc) score = (float)atof(argv[++i]);
        else if (std::strcmp(argv[i], "--iou")   == 0 && i + 1 < argc) iou   = (float)atof(argv[++i]);
        else if (std::strcmp(argv[i], "--headless") == 0)              headless = true;
        else if (std::strcmp(argv[i], "--save")  == 0 && i + 1 < argc) out_path = argv[++i];
        else if (std::strcmp(argv[i], "--help")  == 0) {
            printf("Usage: yolo_camera [--model PATH] [--cam IDX | --video PATH]\n"
                   "                   [--width W] [--height H]\n"
                   "                   [--score T] [--iou T] [--headless] [--save PATH]\n");
            return 0;
        }
    }

    setbuf(stdout, nullptr);   // immediate flush — useful when run headless / logged
    setbuf(stderr, nullptr);
    print_gpu_info();
    printf("Loading TorchScript model: %s\n", model.c_str());

    TorchScriptPipeline pipe(model, 640, 640, 80, score, iou);

    cv::VideoCapture cap;
    if (!video_path.empty()) {
        cap.open(video_path);
        printf("Opening video file: %s\n", video_path.c_str());
    } else {
        v4l2_pin_framerate(cam_id);
        cap.open(cam_id, cv::CAP_V4L2);
        if (!cap.isOpened()) cap.open(cam_id);   // fallback to any backend
        // FOURCC must be set BEFORE width/height or V4L2 silently keeps YUYV
        // (which caps at 10 FPS @ 720p on most laptop webcams; MJPG runs 30 FPS).
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  cam_w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
        cap.set(cv::CAP_PROP_FPS, 30);
        int fourcc = (int)cap.get(cv::CAP_PROP_FOURCC);
        char fcc[5] = {(char)(fourcc & 0xff), (char)((fourcc>>8) & 0xff),
                       (char)((fourcc>>16) & 0xff), (char)((fourcc>>24) & 0xff), 0};
        printf("Opening camera /dev/video%d at %dx%d (fourcc=%s, %.0f fps)\n",
               cam_id, cam_w, cam_h, fcc, cap.get(cv::CAP_PROP_FPS));
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open video source\n");
        return 1;
    }

    cv::VideoWriter writer;
    bool will_save = !out_path.empty();

    // EMA for displayed FPS
    float ema_fps = 0.0f;
    int frame_count = 0;
    auto t_prev = std::chrono::steady_clock::now();

    cv::Mat frame;
    float ema_read = 0.0f, ema_write = 0.0f, ema_draw = 0.0f;
    while (true) {
        auto t_read0 = std::chrono::steady_clock::now();
        bool ok = cap.read(frame);
        float read_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_read0).count();
        ema_read = ema_read == 0.0f ? read_ms : 0.9f * ema_read + 0.1f * read_ms;
        if (!ok || frame.empty()) {
            if (!video_path.empty()) break;   // end of video file
            continue;
        }
        // Ensure contiguous BGR 8UC3
        if (frame.type() != CV_8UC3) {
            cv::Mat tmp;
            cv::cvtColor(frame, tmp, cv::COLOR_BGRA2BGR);
            frame = tmp;
        }

        auto dets = pipe.infer_raw(frame.data, frame.cols, frame.rows, /*is_bgr=*/true);
        auto t = pipe.last_timings();

        auto t_draw0 = std::chrono::steady_clock::now();
        for (const auto& d : dets) draw_box(frame, d);
        float draw_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_draw0).count();
        ema_draw = ema_draw == 0.0f ? draw_ms : 0.9f * ema_draw + 0.1f * draw_ms;

        auto t_now = std::chrono::steady_clock::now();
        float dt_ms = std::chrono::duration<float, std::milli>(t_now - t_prev).count();
        t_prev = t_now;
        float inst = dt_ms > 0.0f ? 1000.0f / dt_ms : 0.0f;
        ema_fps = ema_fps == 0.0f ? inst : 0.9f * ema_fps + 0.1f * inst;

        char hud[256];
        snprintf(hud, sizeof(hud),
                 "FPS %.1f | read %.1f infer %.1f draw %.1f write %.1f ms | %zu dets",
                 ema_fps, ema_read, t.total, ema_draw, ema_write, dets.size());
        cv::putText(frame, hud, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0, 255, 0}, 2);

        if (will_save) {
            if (!writer.isOpened()) {
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open(out_path, fourcc, 30.0, frame.size());
                if (!writer.isOpened()) {
                    fprintf(stderr, "Failed to open writer for %s\n", out_path.c_str());
                    will_save = false;
                }
            }
            if (writer.isOpened()) {
                auto t_w0 = std::chrono::steady_clock::now();
                writer.write(frame);
                float w_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - t_w0).count();
                ema_write = ema_write == 0.0f ? w_ms : 0.9f * ema_write + 0.1f * w_ms;
            }
        }

        if (!headless) {
            cv::imshow("YOLOv8-CUDA (q/ESC to quit)", frame);
            int k = cv::waitKey(1) & 0xFF;
            if (k == 27 || k == 'q') break;
        } else {
            ++frame_count;
            if (frame_count % 10 == 0) {
                printf("[%d] fps=%.1f  read=%.1f  infer=%.1f  draw=%.1f  write=%.1f  dets=%zu\n",
                       frame_count, ema_fps, ema_read, t.total, ema_draw, ema_write, dets.size());
            }
            if (frame_count >= 60) break;   // ~2s of capture for smoke test
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    if (!headless) cv::destroyAllWindows();
    return 0;
}
