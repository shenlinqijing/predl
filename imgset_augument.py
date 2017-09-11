from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import six
import six.moves as sm
import time
import os

np.random.seed(int(time.time()))
ia.seed(int(time.time()))

def image_resize(image):
    shape = image.shape
    bs=300
    if shape[0] > bs and shape[1] > bs:
        if shape[0] > shape[1]:
            width = bs
            height = int(bs * (float(shape[0]) / float(shape[1])))
        else:
            height = bs
            width = int(bs * (float(shape[1]) / float(shape[0])))
        image = misc.imresize(image, (height, width))
        return image
    else:
        return image


def single_image_augument(path):
    image_set=[]
    image=misc.imread(path)
    image=image_resize(image)
    augumentors=[
        ("NOOP",iaa.Noop()),
        ("Fliplr", iaa.Fliplr(1)),
        ("Flipud", iaa.Flipud(1)),
        ("Crop", iaa.Crop(percent=(0.1, 0.15))),
        ("Blur", iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.5, 0.8)),  # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(3,5)),  # blur image using local means with kernel sizes between 2 and 7
            iaa.MedianBlur(k=(3,5)),  # blur image using local medians with kernel sizes between 2 and 7
        ])),
        ("sharpen", iaa.Sharpen(alpha=(0.3, 0.5), lightness=(0.8, 1.3))),  # sharpen images
        ("multipy", iaa.Multiply((0.4, 0.7), per_channel=True)),  # change brightness of images (50-150% of original value)
        ("contrast", iaa.ContrastNormalization((0.70, 1.60), per_channel=True)),  # improve or worsen the contrast
        ("gray", iaa.Grayscale(alpha=(0.5, 1.0))),
        ("GaussianNoise", iaa.AdditiveGaussianNoise(scale=0.10 * 255, per_channel=True)),  # add gaussian noise to images
        ("rotate",iaa.Affine(rotate=(30, 50))),
        ("shear",iaa.Affine(shear=(12, 18))),
        #("Piecewise",iaa.PiecewiseAffine(scale=(0.01, 0.02))),

    ]
    for name,aug in augumentors:
        aug_image=aug.augment_image(image)
        image_set.append((name,aug_image))
    return image_set



#make sure "desFolder" folder existed!!if not ,create it!!!!!!
def main():
    srcFolder="/home/dandan/imgset/raw/cold/"
    desFolder="/home/dandan/imgset/amt/cold/"
    srcImages=os.listdir(srcFolder)


    for srcImage in srcImages:
        preName = srcImage.split(".")[0]
        fileType= srcImage.split(".")[1]
        srcPath=srcFolder+srcImage
        try:
            results = single_image_augument(srcPath)
        except:
            errmsg="process %s error!!!\n"%srcPath
            print (errmsg)
            f=open("err.log", "a")
            f.write(errmsg)
            f.close()
            continue
        for augName,data in results:
            resName=desFolder+"%s_%s.%s"%(preName,augName,fileType)
            misc.imsave(resName, data)
            print ("save %s done~"%resName)

if __name__ == "__main__":
    main()

