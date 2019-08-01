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


def get_crops_from_bounding_boxes(bounding_boxes, original_image):
    bounding_boxes = np.asarray(bounding_boxes)

    bbox_classes = bounding_boxes[:, 5]
    tvmonitor = (bbox_classes == 62)
    laptop = (bbox_classes == 63)
    
    cropped_images = []
    for bbox in bounding_boxes[tvmonitor | laptop]:
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]

        x1 = int(max(0, bbox[0] - 0.1 * width))
        y1 = int(max(0, bbox[1] - 0.1 * height))
        x2 = int(min(bbox[2] + 0.1 * width, original_image.shape[1]))
        y2 = int(min(bbox[3] + 0.1 * height, original_image.shape[0]))

        cropped = original_image[y1:y2, x1:x2]
        cropped_images.append(cropped)
    return cropped_images


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
image_path      = (len(sys.argv) == 2 and sys.argv[1]) or "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

response = requests.post(
    SERVER_URL,
    data=json.dumps({
        'instances': image_data.tolist(),
    }),
)
response.raise_for_status()
predictions = response.json()['predictions'][0]


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
