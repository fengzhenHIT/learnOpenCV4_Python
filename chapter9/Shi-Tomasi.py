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
    # 将图像进行拷贝便于使用cv.drawKeypoints()函数绘制角点
    image1 = image.copy()

    # 转为灰度图像，并将数据类型转换为float32
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 检测Shi-Tomasi角点
    corners = cv.goodFeaturesToTrack(gray, 500, 0.01, 0.04)
    corners = np.int0(corners)

    # 绘制角点（Mode 1：使用cv.circle()函数）
    kps = []
    for corner in corners:
        x, y = corner.ravel()
        cv.circle(image, (x, y), 3, (0, 255, 255), -1)

        # 将角点转化为KeyPoint类，以便于方法2的绘制
        kps.append(cv.KeyPoint(x, y, 1))

    # 绘制角点（Mode 2：使用cv.drawKeypoints()函数）
    result = cv.drawKeypoints(image1, kps, None, (), cv.DRAW_MATCHES_FLAGS_DEFAULT)

    # 展示结果
    cv.imshow('Shi-Tomasi KeyPoints(Mode 1)', image)
    cv.imshow('Shi-Tomasi KeyPoints(Mode 2)', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
