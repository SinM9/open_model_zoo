// Copyright (C)

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/objdetect/face.hpp>

#include <iostream>
#include <vector>
#include <string>

void visualize(cv::Mat& input, cv::Mat& faces, double fps, int thickness = 2)
{
    for (int i = 0; i < faces.rows; i++)
    {
        // Draw bounding box
        rectangle(input, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0), thickness);
        // Put score
        std::string score = cv::format("%.2f", faces.at<float>(i, 14));
        putText(input, score, cv::Point(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1))+12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        // Draw landmarks
        circle(input, cv::Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), thickness);
        circle(input, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), thickness);
    }
    // Put fps
    std::string fpsString = cv::format("FPS : %.2f", (float)fps);    
    putText(input, fpsString, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}


int main(int argc, char** argv)
{
    const cv::String keys =
        "{help h usage ?  |                          | YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection). }"
        "{input i         |                          | Path to the input image or video. Omit for using default camera.}"
        "{model m         |                          | Path to the model. You can download model "
                                                       "from https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx}"
        "{backend b       | dnn::DNN_BACKEND_OPENCV  | Select a computation backend: "
                                                          "dnn::DNN_BACKEND_OPENCV, "
                                                          "dnn::DNN_BACKEND_CUDA } "
        "{target t        | dnn::DNN_TARGET_CPU      | Select a target device: "
                                                          "dnn::DNN_TARGET_CPU, "
                                                          "dnn::DNN_TARGET_CUDA, "
                                                          "dnn::DNN_TARGET_CUDA_FP16 }"                                                
        "{conf_threshold  | 0.9                      | Filter out faces of score < score_threshold}"
        "{nms_threshold   | 0.3                      | Suppress bounding boxes of iou >= nms_threshold}"
        "{top_k           | 5000                     | Keep top_k bounding boxes before NMS}"
        ;
    
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    cv::String input = parser.get<cv::String>("input"); 
    cv::String modelPath = parser.get<cv::String>("model"); 
    if (!parser.has("model")) {
        std::cerr << "Please set path to the model" << std::endl;
        return -1;
    }

    float confThreshold = parser.get<float>("conf_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    cv::String config = "";
    cv::Size inputSize = {320, 320};

    cv::Ptr<cv::FaceDetectorYN> model = cv::FaceDetectorYN::create(modelPath, config, inputSize, 
                                        confThreshold, nmsThreshold, topK, backendId, targetId);
    
    cv::TickMeter tm;
    if (parser.has("input")) {
        if (cv::imread(cv::samples::findFile(input)).empty()) {
            // video            
            cv::VideoCapture cap(input);
            int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            model->setInputSize({w, h});

            while (cv::waitKey(1) < 0) {
                cv::Mat frame;
                if (!cap.read(frame)) {
                    std::cerr << "Can't grab frame! Stop\n";
                    break;
                }
                cv::Mat faces;
                tm.start();
                model->detect(frame, faces);
                tm.stop();
                
                cv::Mat result = frame.clone();
                visualize(result, faces, tm.getFPS());
                cv::imshow("YuNet Demo", result);

                tm.reset();
            }
        } else {
            // image
            cv::Mat image = cv::imread(cv::samples::findFile(input));
            model->setInputSize(image.size());
            cv::Mat faces;

            tm.start();
            model->detect(image, faces);
            tm.stop();

            visualize(image, faces, tm.getFPS());
            cv::imshow("YuNet Demo", image);
            cv::waitKey(0);
        }
    } else {
        // camera
        int deviceId = 0;
        cv::VideoCapture cap(deviceId);

        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        model->setInputSize({w, h});

        while (cv::waitKey(1) < 0) {
            cv::Mat frame;
            if (!cap.read(frame))
            {
                std::cerr << "Can't grab frame! Stop\n";
                break;
            }
            cv::Mat faces;
            tm.start();
            model->detect(frame, faces);
            tm.stop();
            
            cv::Mat result = frame.clone();
            visualize(result, faces, tm.getFPS());
            cv::imshow("YuNet Demo", result);

            tm.reset();
        }
    }
    return 0;
}