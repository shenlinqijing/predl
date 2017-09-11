#!/usr/bin/env sh

./build/tools/caffe train \
    -solver  models/vgg/solver.prototxt \
    -weights models/vgg/VGG_ILSVRC_16_layers.caffemodel
