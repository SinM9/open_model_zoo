# Copyright (C) 

import argparse

import numpy as np
import cv2 as cv

backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx', help='Path to the model')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--conf_threshold', type=float, default=0.9, help='Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')

args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(255, 0, 0), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

if __name__ == '__main__':
    # Instantiate YuNet
    model = cv.FaceDetectorYN.create(
                model=args.model,
                config="",
                input_size=[320, 320],
                score_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                top_k=args.top_k,
                backend_id=args.backend,
                target_id=args.target)
    # If input is an image
    if args.input is not None:

        if cv.imread(args.input) is not None:
            image = cv.imread(args.input)
            h, w, _ = image.shape

            # Inference
            model.setInputSize([w, h])
            results = model.detect(image)[1]

            # Draw results on the input image
            image = visualize(image, results)

            # Visualize results
            cv.imshow('YuNet Demo', image)
            cv.waitKey(0)

        else:
            cap = cv.VideoCapture(args.input)
            w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            model.setInputSize([w, h])

            tm = cv.TickMeter()
            while cv.waitKey(1) < 0:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    print('No frames grabbed!')
                    break

                # Inference
                tm.start()
                results = model.detect(frame)[1]
                tm.stop()

                # Draw results on the input image
                frame = visualize(frame, results, fps=tm.getFPS())

                # Visualize results
                cv.imshow('YuNet Demo', frame)

                tm.reset()


    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        model.setInputSize([w, h])

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            results = model.detect(frame)[1]
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())

            # Visualize results
            cv.imshow('YuNet Demo', frame)

            tm.reset()
