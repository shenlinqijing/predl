#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/dandan/DL/caffe/data/sexy/
DES=/home/dandan/DL/caffe/examples/sexy/
TOOLS=/home/dandan/DL/caffe/build/tools/

#echo $TOOLS/compute_image_mean $EXAMPLE/HAT_train_lmdb \
#    $DATA/HAT_mean.binaryproto
$TOOLS/compute_image_mean $EXAMPLE/sexy_train_lmdb \
    $DES/sexy_mean.binaryproto
echo "Done."
