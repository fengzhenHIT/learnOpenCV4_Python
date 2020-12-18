# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    # 设置图像旋转角度、尺寸、旋转中心参数
    angle = 30
    h, w = img.shape[:-1]
    size = (w, h)
    center = (w / 2.0, h / 2.0)
    # 计算仿射变换矩阵
    rotation0 = cv.getRotationMatrix2D(center, angle, 1)
    # 进行仿射变换
    img_warp0 = cv.warpAffine(img, rotation0, size)

    # 根据定义的三个点进行仿射变换
    src_points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1]], dtype='float32')
    dst_points = np.array([[w * 0.11, h * 0.2], [w * 0.15, h * 0.7], [w * 0.81, h * 0.85]], dtype='float32')
    rotation1 = cv.getAffineTransform(src_points, dst_points)
    img_warp1 = cv.warpAffine(img, rotation1, size)

    # 展示结果
    cv.imshow('img_warp0', img_warp0)
    cv.imshow('img_warp1', img_warp1)
    cv.waitKey(0)
    cv.destroyAllWindows()
