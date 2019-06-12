#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from pathlib import Path
from PIL import Image

import base64
import json
import requests
import sys

SERVER_URL = 'http://localhost:18501/v1/models/yolo:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path      = (len(sys.argv) == 2 and sys.argv[1]) or "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

total_time = 0
response = requests.post(
    SERVER_URL,
    data=json.dumps({
        'instances': image_data.tolist(),
    }),
)
response.raise_for_status()
predictions = response.json()['predictions'][0]


#with tf.Session(graph=graph) as sess:
#    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
#        [return_tensors[1], return_tensors[2], return_tensors[3]],
#                feed_dict={ return_tensors[0]: image_data})

pred_bbox = np.concatenate(
    [
        np.reshape(predictions['conv_sbbox'], (-1, 5 + num_classes)),
        np.reshape(predictions['conv_mbbox'], (-1, 5 + num_classes)),
        np.reshape(predictions['conv_lbbox'], (-1, 5 + num_classes)),
    ],
    axis=0,
)

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.save('result_' + Path(image_path).stem + '.jpg')




