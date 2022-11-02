# Copyright (C)

import logging as log
import sys
import argparse
from time import perf_counter
import numpy as np
import cv2 as cv

from openvino.model_zoo.model_api.performance_metrics import put_highlighted_text, PerformanceMetrics
from openvino.model_zoo.model_api.models import OutputTransform

import monitors
from images_capture import open_images_capture
from helpers import resolution

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def build_argparser():
    backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
    targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
    help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
    help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
    
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')    
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('--input', '-i', type=str, default="0", help='An input to process. The input must be a single image,\
                     a folder of images, video file or camera id. Omit for using default camera.')
    args.add_argument('--model', '-m', type=str, default='', required=True, help='Required. Path to the model')
    args.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
    args.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
    
    common_model_args = parser.add_argument_group('Common model options')
    common_model_args .add_argument('--conf_threshold', type=float, default=0.9, help='Filter out faces of confidence < conf_threshold.')
    common_model_args .add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
    common_model_args .add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    
    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

def visualize(image, results, box_color=(0, 255, 0), text_color=(255, 0, 0), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.2f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

def print_raw_results(faces, frame_id):
    log.debug('  -------------------------- Frame # {} --------------------------  '.format(frame_id))
    if faces is not None:
        for idx, face in enumerate(faces) if faces is not None else []:
            log.debug('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'
                        .format(idx, face[0], face[1], face[2], face[3], face[-1]))

def main():
    # Parsing input args
    args = build_argparser().parse_args()

    log.info('\tOpenCV version: {}'.format(cv.__version__))

    # Preparing Input
    cap = open_images_capture(args.input, args.loop)
    delay = int(cap.get_type() in {'VIDEO', 'CAMERA'})

    model = cv.FaceDetectorYN.create(
                model=args.model,
                config="",
                input_size=[320, 320],
                score_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                top_k=args.top_k,
                backend_id=args.backend,
                target_id=args.target)
    log.info('Reading model {}'.format(args.model))

    metrics = PerformanceMetrics()
    video_writer = cv.VideoWriter()
    ESC_KEY = 27
    key = -1
    frame_number = 0

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            raise RuntimeError("Can't read an image from the input")

        output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
        if args.output_resolution:
            output_resolution = output_transform.new_resolution
           
        else:
            output_resolution = (frame.shape[1], frame.shape[0])
        
        frame = cv.resize(frame, output_resolution)

        presenter = monitors.Presenter(args.utilization_monitors, 55,
                                      (round(output_resolution[0] / 4),
                                       round(output_resolution[1] / 8)))
        
        if args.output and not video_writer.open(args.output, cv.VideoWriter_fourcc(*'MJPG'),
                                                cap.fps(), args.output_resolution):
            raise RuntimeError("Can't open video writer")

        # Inference
        model.setInputSize([frame.shape[1], frame.shape[0]])
        results = model.detect(frame)[1]
        
        # Draw results on the input image
        frame = visualize(frame, results)

        presenter.drawGraphs(frame)        

        if delay or args.no_show:
            metrics.update(start_time, frame)

        if args.raw_output_message:
            print_raw_results(results, frame_number)

        if video_writer.isOpened() and (args.output_limit <= 0 or frame_number <= args.output_limit-1):
            video_writer.write(frame)

        if not args.no_show:
            # Visualize results
            cv.imshow('Face Detection YuNet Demo', frame)
            key = cv.waitKey(delay)
            # Processing keyboard events
            # Pause
            if key in {ord('p'), ord('P'),  ord('0'),  ord(' ')}:
                key = cv.waitKey(0)
            # Quit
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)

        frame_number += 1

    if delay or args.no_show:
        metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

if __name__ == '__main__':
    sys.exit(main() or 0)
