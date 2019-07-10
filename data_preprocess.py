# -*- coding: utf-8 -*-

#Canny
import cv2 as cv
import os
import numpy as np
from PIL import Image
from skimage import data,filters,color
import matplotlib.pyplot as plt


from_path="Train/HR/"
data_path="imgs/train/data"
label_path="imgs/train/label"
def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)
    return edge_output
    #cv.imshow("Canny Edge", edge_output)
    #dst = cv.bitwise_and(image, image, mask= edge_output)
    #return dst
    #cv.imshow("Color Edge", dst)
files=os.listdir(from_path)
index=1
for file in files:
    print('processing： %s' % index)
    img_path=from_path+file
    print(img_path)
    img = cv.imread(img_path)
    crop_size = (480, 320)
    img_new = cv.resize(img, crop_size, interpolation = cv.INTER_CUBIC)
    cv.imwrite(data_path+'/'+file,img_new)

    img1=edge_demo(img_new)
    cv.imwrite(label_path+'/'+file,img1)
    print('succeed： %s' % index)
    
    index += 1
