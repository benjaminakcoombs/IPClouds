import numpy as np
import pandas as pd
import urllib
import requests
import datetime
import cv2
import os
import shutil
import math
from matplotlib import pyplot as plt



def create_mask(img, num):
    
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)    
    rect_value = 940
    canny_thresh = 30
    img_size = [640,480]
        
    rect = (0, 0, len(img[0]), rect_value)

    edges = cv2.Canny(img,canny_thresh,canny_thresh)
    
    _,thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    rect=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh,rect,iterations = 4)
    erosion = cv2.erode(dilation, rect, iterations=3)
    plt.imshow(erosion)

    return erosion


path = r"/vol/bitbucket/bac21/steak-generation/fullRender/fullRender/final"
for folder in ['fullRender_test', 'fullRender_val', 'fullRender_train']:
    full_path = os.path.join(path, folder)
    for parent, directories, files in os.walk(full_path):
        for directory in directories:
            if directory not in ['rgb', 'mask', 'pose']:
                full_path_spec = os.path.join(full_path, directory)
                image_path_1 = os.path.join(full_path_spec, "rgb",  '1.jpg')
                image_path_2 = os.path.join(full_path_spec, "rgb", '4.jpg')
                if not os.path.exists(os.path.join(full_path_spec, "mask")):
                    os.mkdir(os.path.join(full_path_spec, "mask"))
                image1 = cv2.imread(image_path_1)
                image2 = cv2.imread(image_path_2)
                mask1 = create_mask(image1, 4)
                mask2 = create_mask(image2, 4)
                image1 = cv2.resize(image1, (640, 480))
                image2 = cv2.resize(image2, (640, 480))
                mask1 = cv2.resize(mask1, (640, 480))
                mask2 = cv2.resize(mask2, (640, 480))
                cv2.imwrite(os.path.join(full_path_spec, "mask", '1.jpg'), mask1)
                cv2.imwrite(os.path.join(full_path_spec, "mask", '4.jpg'), mask2)
