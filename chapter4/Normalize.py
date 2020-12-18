# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 对数组进行归一化
    data = np.array([2.0, 8.0, 10.0])
    # 绝对值求和归一化
    data_L1 = cv.normalize(data, None, 1.0, 0.0, cv.NORM_L1)
    # 模长归一化
    data_L2 = cv.normalize(data, None, 1.0, 0.0, cv.NORM_L2)
    # 最大值归一化
    data_Inf = cv.normalize(data, None, 1.0, 0.0, cv.NORM_INF)
    # 偏移归一化
    data_L2SQR = cv.normalize(data, None, 1.0, 0.0, cv.NORM_MINMAX)
    # 展示结果
    print('绝对值求和归一化结果为：\n{}'.format(data_L1))
    print('模长归一化结果为：\n{}'.format(data_L2))
    print('最大值归一化结果为：\n{}'.format(data_Inf))
    print('偏移归一化结果为：\n{}'.format(data_L2SQR))

    # 对图像直方图进行归一化
    # 读取图像
    image = cv.imread('./images/apple.jpg')
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read apple.jpg.')
        sys.exit()

    # 将图像转为灰度图像
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 对图像进行直方图计算
    hist_item = cv.calcHist([gray_image], [0], None, [256], [0, 256])

    # 对直方图进行绝对值求和归一化
    image_L1 = cv.normalize(hist_item, None, 1, 0, cv.NORM_L1)
    # 对直方图进行最大值归一化
    image_Inf = cv.normalize(hist_item, None, 1, 0, cv.NORM_INF)
    # 展示结果
    plt.plot(image_L1)
    plt.show()
    plt.plot(image_Inf)
    plt.show()
