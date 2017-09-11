#! /usr/bin/env python
#coding=utf-8
import numpy as np
import sys,os
# 设置当前的工作环境在caffe下
caffe_root = '/home/dandan/DL/caffe/' 
# 我们也把caffe/python也添加到当前环境
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)#更换工作目录

# 设置网络结构
#net_file=caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
net_file=caffe_root + 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
# 添加训练之后的参数
#caffe_model=caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
caffe_model=caffe_root + 'models/vgg/vgg_sexy_train_iter_15000.caffemodel'
# 均值文件
mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

# 这里对任何一个程序都是通用的，就是处理图片
# 把上面添加的两个变量都作为参数构造一个Net
net = caffe.Net(net_file,caffe_model,caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换

# channel 放到前面
transformer.set_transpose('data', (2,0,1))

#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_mean('data', np.array([104, 117, 123]))
# 图片像素放大到[0-255]
transformer.set_raw_scale('data', 255) 
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2,1,0))

# 这里才是加载图片
im=caffe.io.load_image(caffe_root+'examples/images/5.jpg')
# 用上面的transformer.preprocess来处理刚刚加载图片
net.blobs['data'].data[...] = transformer.preprocess('data',im)

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

#注意,网络开始向前传播啦
out = net.forward()

print net.blobs['prob'].data[0]

result=net.blobs['prob'].data[0].flatten()
print ("sexy:%.2f"%result[0])
print ("norm:%.2f"%result[1])

#imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
#labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-2:-1]
#for i in np.arange(top_k.size):
#    print top_k[i] , net.blobs['prob'].data[0][top_k[i]]
