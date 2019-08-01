#!/bin/bash
set -eux

MODELS_ROOT=${MODELS_ROOT:-"/tmp"}
echo $MODELS_ROOT

# download saved model
mkdir -p checkpoint
if [[ ! -f checkpoint/yolov3_coco.ckpt.meta ]]; then
    curl -L https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz | tar -xvzC checkpoint/
fi

# convert model from darkflow to tf
python convert_weight.py

# convert checkpoint to saved_model
python freeze_graph_serve.py "$MODELS_ROOT/yolo/1"
