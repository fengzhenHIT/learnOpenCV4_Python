# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像
    image = cv.imread('./images/lena.jpg')
    if image is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 计算Harris系数
    harris = cv.cornerHarris(gray, 2, 3, 0.04, borderType=cv.BORDER_DEFAULT)

    # 对Harris进行归一化便于进行数值比较
    harris_nor = cv.normalize(harris, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    harris_nor = harris_nor.astype('uint8')

    # 寻找Harris角点
    kps = []
    for i in np.argwhere(harris_nor > 125):
        kps.append(cv.KeyPoint(i[1], i[0], 1))

    # 绘制角点
    result = cv.drawKeypoints(image, kps, None)

    # 展示结果
    cv.imshow('R', harris_nor)
    cv.imshow('Harris KeyPoints', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
