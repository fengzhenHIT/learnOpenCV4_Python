# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 读取图像
    image1 = cv.imread('./images/Hist_Match.png')
    image2 = cv.imread('./images/equalLena.png')
    # 判断图片是否读取成功
    if image1 is None or image2 is None:
        print('Failed to read Hist_Match.png or equalLena.png.')
        sys.exit()

    # 计算两张图像的直方图
    hist_image1 = cv.calcHist([image1], [0], None, [256], [0, 256])
    hist_image2 = cv.calcHist([image2], [0], None, [256], [0, 256])
    # 对直方图进行归一化
    hist_image1 = cv.normalize(hist_image1, None, norm_type=cv.NORM_L1)
    hist_image2 = cv.normalize(hist_image2, None, norm_type=cv.NORM_L1)

    # 计算两张图像直方图的累计概率
    hist1_cdf = np.zeros((256, ))
    hist2_cdf = np.zeros((256, ))
    hist1_cdf[0] = 0
    hist2_cdf[0] = 0
    for i in range(1, 256):
        hist1_cdf[i] = hist1_cdf[i - 1] + hist_image1[i]
        hist2_cdf[i] = hist2_cdf[i - 1] + hist_image2[i]

    # 构建累计概率误差矩阵
    diff_cdf = np.zeros((256, 256))
    for k in range(256):
        for j in range(256):
            diff_cdf[k][j] = np.fabs((hist1_cdf[k] - hist2_cdf[j]))

    # 生成LUT映射表
    lut = np.zeros((256, ), dtype='uint8')
    for m in range(256):
        # 查找源灰度级为i的映射灰度和i的累计概率差值最小的规定化灰度
        min_val = diff_cdf[m][0]
        index = 0
        for n in range(256):
            if min_val > diff_cdf[m][n]:
                min_val = diff_cdf[m][n]
                index = n
        lut[m] = index
    result = cv.LUT(image1, lut)

    # 展示结果
    cv.imshow('Origin Image1', image1)
    cv.imshow('Origin Image2', image2)
    cv.imshow('Result', result)
    _, _, _ = plt.hist(x=image1.ravel(), bins=256, range=[0, 256])
    plt.show()
    _, _, _ = plt.hist(x=image2.ravel(), bins=256, range=[0, 256])
    plt.show()
    _, _, _ = plt.hist(x=result.ravel(), bins=256, range=[0, 256])
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()
