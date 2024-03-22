import os

import cv2
import numpy as np
import cupy as cp
import time
from scipy.interpolate import interpn

N = 1
N_add = 3  # 多生成的帧
dir_file='./image_interpolation/'
image_path = './image_split/'
#file_name = 'test'

def video_interpolation():
    idex = 1
    image_num=len(os.listdir(image_path))
    for i in range(1,image_num):
        image1 = image_path + str(i) + '.png'
        image2 = image_path + str(i + 1) + '.png'
        img1 = cv2.imread(image1)
        img1_1 = cp.asarray(img1)
        img2 = cv2.imread(image2)
        img2_2 = cp.asarray(img2)
        width, height = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
        img3 = cp.zeros((N + N_add, width, height, 3))
        for k in range(0, N + N_add):  # 生成中间帧
            t = k / (N_add + N) # t=0-->img1 t= N+N_add-->img2
            start = time.time()
            for x in range(width):
                for y in range(height):
                    img3[k] = (1 - t) * img1_1 + t * img2_2  # 线性插值公式
            end = time.time()
            print(end - start, idex)
            img4 = cp.asnumpy(img3)
            cv2.imwrite(dir_file + str(idex) + '.png', img4[k])
            idex += 1


video_interpolation()