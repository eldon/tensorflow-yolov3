#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from core.yolov3 import YOLOV3

pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=(1, 416, 416, 3))


model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

tf.saved_model.simple_save(
    sess,
    '/tmp/yolo/1/',
    inputs={'input_data': input_data},
    outputs={
        'conv_sbbox': tf.get_default_graph().get_tensor_by_name('pred_sbbox/concat_2:0'),
        'conv_mbbox': tf.get_default_graph().get_tensor_by_name('pred_mbbox/concat_2:0'),
        'conv_lbbox': tf.get_default_graph().get_tensor_by_name('pred_lbbox/concat_2:0'),
    },
)

