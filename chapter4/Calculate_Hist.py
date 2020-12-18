# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np

# 设置不显示科学计数法，显示普通数字
np.set_printoptions(suppress=True)


if __name__ == '__main__':
    # 以灰度方式读取图像
    image = cv.imread('./images/apple.jpg', 0)
    # 判断是否读取成功
    if image is None:
        print("Failed to read apple.jpg.")
        sys.exit()
    # 对图像进行直方图计算
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    # 输出结果
    print('统计灰度直方图为：\n{}'.format(hist))
