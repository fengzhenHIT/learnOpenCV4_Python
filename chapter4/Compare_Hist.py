# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def normalize_image(path):
    # 以灰度方式读取图像
    image = cv.imread(path, 0)
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read image.')
        sys.exit()
    # 绘制直方图（可省略）
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(path.split('/')[-1])
    plt.show()
    # 计算图像直方图
    image_hist = cv.calcHist([image], [0], None, [256], [0, 256])
    # 进行归一化
    normalize_result = np.zeros(image_hist.shape, dtype=np.float32)
    cv.normalize(image_hist, dst=normalize_result, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
    return normalize_result


def compare_hist(image1_path, image2_path):
    image1 = normalize_image(image1_path)
    image2 = normalize_image(image2_path)
    # 进行图像直方图比较
    return round(cv.compareHist(image1, image2, method=cv.HISTCMP_CORREL), 2)


if __name__ == '__main__':
    img1_path = './images/Compare_Hist_1.jpg'
    img2_path = './images/Compare_Hist_2.jpg'
    img3_path = './images/Compare_Hist_3.jpg'
    img4_path = './images/Compare_Hist_4.jpg'

    print('Compare_Hist_1.jpg与Compare_Hist_2.jpg的相似性为：%s' % (compare_hist(img1_path, img2_path)))
    print('Compare_Hist_3.jpg与Compare_Hist_4.jpg的相似性为：%s' % (compare_hist(img3_path, img4_path)))
