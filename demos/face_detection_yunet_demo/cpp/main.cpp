/*
    Copyright (C)
*/

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <utils/common.hpp>
#include <utils/images_capture.h>

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "Required. Path to the model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] =
    "An input to process. The input must be a single image, a folder of images,"
    " video file or camera id. Omit for using default camera.";
DEFINE_string(i, "0", i_msg);

constexpr char b_msg[] = "Select a computation backend: OPENCV (default), CUDA";
DEFINE_string(b, "OPENCV", b_msg);

constexpr char d_msg[] = "Select a target device: CPU (default), CUDA, CUDA_FP16";
DEFINE_string(d, "CPU", d_msg);

constexpr char conf_msg[] = "Filter out faces of score < score_threshold";
DEFINE_double(conf_threshold, 0.9, conf_msg);

constexpr char nms_msg[] = "Suppress bounding boxes of iou >= nms_threshold";
DEFINE_double(nms_threshold, 0.3, nms_msg);

constexpr char top_k_msg[] = "Keep top_k bounding boxes before NMS";
DEFINE_uint32(top_k, 5000, top_k_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char utilization_monitors_msg[] = "Optional. List of monitors to show initially.";
DEFINE_string(u, "", utilization_monitors_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char output_resolution_msg[] =
    "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";
DEFINE_string(output_resolution, "", output_resolution_msg);

constexpr char output_limit_msg[] = "Optional. Number of frames to store in output. "
                                    "If 0 is set, all frames are stored.";
DEFINE_uint32(output_limit, 1000, output_limit_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[ -b]                                         " << b_msg
                  << "\n\t[ -d]                                         " << d_msg
                  << "\n\t[ -conf_threshold <NUMBER>]                   " << conf_msg
                  << "\n\t[ -nms_threshold <NUMBER>]                    " << nms_msg
                  << "\n\t[ -top_k <NUMBER>]                            " << top_k_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--show] ([--noshow])                         " << show_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[ --output_resolution]                        " << output_resolution_msg
                  << "\n\t[ --output_limit <NUMBER>]                    " << output_limit_msg
                  << "\n\t[ -u]                                         " << utilization_monitors_msg                  
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause\n";
        showAvailableDevices();
        std::cout << ov::get_openvino_version() << std::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    } if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    } if (!FLAGS_d.empty() && !(FLAGS_d == "CPU" || FLAGS_d == "CUDA" || FLAGS_d == "CUDA_FP16")) {
        throw std::invalid_argument{"-d must be value from list: CPU (default), CUDA, CUDA_FP16"};
    } if (!FLAGS_b.empty() && !(FLAGS_b == "OPENCV" || FLAGS_b == "CUDA")) {
        throw std::invalid_argument{"-b must be value from list: OPENCV (default), CUDA"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
    slog::info << "OpenCV version: " << CV_VERSION << slog::endl;
}
} // namespace

void visualize(cv::Mat& input, cv::Mat& faces, int thickness = 2) {
    for (int i = 0; i < faces.rows; ++i) {
        // Draw bounding box
        rectangle(input, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0), thickness);
        // Put score
        std::string score = cv::format("%.f", faces.at<float>(i, 14) * 100) + '%';
        putHighlightedText(input, score, cv::Point(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1))-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        // Draw landmarks
        circle(input, cv::Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), thickness);
    }
}

void printRawResults(cv::Mat& faces, int frame_id) {
    slog::info << "  -------------------------- Frame " << frame_id << " --------------------------  "  << slog::endl;
    if (!faces.empty()) {
        for (int i = 0; i < faces.rows; ++i) {
            slog::info << "Face " << i
                << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
                << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
                << "score: " << cv::format("%.2f", faces.at<float>(i, 14))
                << slog::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::set_terminate(catcher);

    // Parsing and validation of input args
    parse(argc, argv);

    // Preparing Input
    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

    std::map<std::string, int> devices{{"CPU", cv::dnn::DNN_TARGET_CPU}, {"CUDA", cv::dnn::DNN_TARGET_CUDA}, {"CUDA_FP16", cv::dnn::DNN_TARGET_CUDA_FP16}};
    std::map<std::string, int> backends{{"OPENCV", cv::dnn::DNN_BACKEND_OPENCV}, {"CUDA", cv::dnn::DNN_BACKEND_CUDA}};

    cv::Ptr<cv::FaceDetectorYN> model = cv::FaceDetectorYN::create(FLAGS_m, "", {320, 320}, 
                                            static_cast<float>(FLAGS_conf_threshold), 
                                            static_cast<float>(FLAGS_nms_threshold),
                                            FLAGS_top_k, backends[FLAGS_b], devices[FLAGS_d]);
    slog::info << "Target device: " << FLAGS_d << slog::endl;
    slog::info << "Computation backend: " << FLAGS_b << slog::endl;
    slog::info << "Reading model " << FLAGS_m << slog::endl;
    PerformanceMetrics metrics;
    Presenter presenter(FLAGS_u);
    LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_output_limit};
    cv::Size outputResolution;
    OutputTransform outputTransform = OutputTransform();
    size_t found = FLAGS_output_resolution.find("x");

    int delay = 0;
    if (cap->getType()=="VIDEO" || cap->getType()=="CAMERA") {
        delay = 1;
    }
    int frame_number = 0;

    while (1) {
        auto time = std::chrono::steady_clock::now();

        cv::Mat input_frame = cap->read();
        if (input_frame.empty()) {
            slog::info << "Can't read an image from the input" << slog::endl;
            break;
        }
        if (found == std::string::npos) {
            outputResolution = input_frame.size();
        } else {
             outputResolution =
                    cv::Size{std::stoi(FLAGS_output_resolution.substr(0, found)),
                             std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
            outputTransform = OutputTransform(input_frame.size(), outputResolution);
            outputResolution = outputTransform.computeResolution();        
        }

        cv::Mat frame;
        cv::resize(input_frame, frame, outputResolution);

        // Inference
        model->setInputSize(frame.size());
        cv::Mat faces;
        model->detect(frame, faces);

        // Draw results on the input image
        visualize(frame, faces);

        presenter.drawGraphs(frame);

        if (delay && FLAGS_show) {
            metrics.update(time, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
        }
        if (FLAGS_r) {
            printRawResults(faces, frame_number);
        }
        videoWriter.write(frame);

        if (FLAGS_show) {
            // Visualize results
            cv::imshow("Face Detection YuNet Demo", frame);
            int key = cv::waitKey(delay);
            // Processing keyboard events
            // Pause
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            // Quit
            if (27 == key || 'Q' == key || 'q' == key) {
                break;
            }
            presenter.handleKey(key);
        }
        frame_number++;
    }
    if (delay && FLAGS_show) {
        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
    }
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}