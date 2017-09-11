#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirsset -e

DB_ROOT=/home/dandan/DL/caffe/data/sexy/
LIST_ROOT=/home/dandan/imgset/
TOOLS=/home/dandan/DL/caffe/build/tools/
VAL_DATA_ROOT=/home/dandan/imgset/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE_HEIGHT=300
RESIZE_WIDTH=300

echo "Creating train  lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $LIST_ROOT/val.txt \
    $DB_ROOT/sexy_val_lmdb


