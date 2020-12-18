# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/dial.png')
    if img is None:
        print('Failed to read dial.png.')
        sys.exit()

    h, w = img.shape[:-1]
    # 计算极坐标在图像中的原点
    center = (w / 2, h / 2)
    # 正极坐标变换
    img_res = cv.warpPolar(img, (300, 600), center, center[0], cv.INTER_LINEAR + cv.WARP_POLAR_LINEAR)
    # 逆极坐标变换
    img_res1 = cv.warpPolar(img_res, (w, h), center, center[0], cv.INTER_LINEAR + cv.WARP_POLAR_LINEAR + cv.WARP_INVERSE_MAP)

    # 展示结果
    cv.imshow('Origin', img)
    cv.imshow('img_res', img_res)
    cv.imshow('img_res1', img_res1)
    cv.waitKey(0)
    cv.destroyAllWindows()
