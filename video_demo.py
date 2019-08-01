#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import argparse
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tqdm import trange


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "./docs/images/road.mp4"
# video_path      = 0
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)


def make_vid_output():
    with tf.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)
        while True:
            return_value, frame = vid.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                raise ValueError("No image!")
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break


def export_yolo_video(video_path, output_path):
    vid = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(
        str(Path(output_path, Path(video_path).stem + '_output.mp4')),
        cv2.VideoWriter_fourcc(*'mp4v'),
        float(vid.get(cv2.CAP_PROP_FPS)),
        (round(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    outputs = []
    print(Path(output_path, Path(video_path).stem + '_output.mp4'))
    print(Path(output_path, Path(video_path).stem + '_output.npz'))
    max_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    with tf.Session(graph=graph) as sess:
        for _ in trange(int(max_frames)):
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("No image!")
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            outputs.append(bboxes)

            image = utils.draw_bbox(frame, bboxes)

            result = np.asarray(image)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(result)

    vid.release()
    out.release()
    np.savez(str(Path(output_path, Path(video_path).stem + '_output.npz')), outputs)

if __name__ == '__main__':
    import sys
    print('*****************************')
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
        print(export_yolo_video(video_path, output_path))
        exit(0)
    print(make_vid_output())

