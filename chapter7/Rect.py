# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np


if __name__ == '__main__':
    # 读取图像stuff.jpg
    image = cv.imread('./images/stuff.jpg')
    if image is None:
        print('Failed to read stuff.jpg.')
        sys.exit()
    cv.imshow('Origin', image)

    # 提取图像边缘
    canny = cv.Canny(image, 80, 160, 3)
    cv.imshow('Canny Image', canny)

    # 膨胀运算
    kernel = cv.getStructuringElement(0, (3, 3))
    canny = cv.dilate(canny, kernel=kernel)

    # 轮廓检测及绘制
    contours, hierarchy = cv.findContours(canny, mode=0, method=2)

    # 寻找并绘制轮廓外接矩形
    img1 = image.copy()
    img2 = image.copy()
    for i in range(len(contours)):
        # 绘制轮廓的最大外接矩形
        max_rect = cv.boundingRect(contours[i])
        cv.rectangle(img1, max_rect, (0, 0, 255), 2, 8, 0)
        # 绘制轮廓的最小外接矩形
        min_rect = cv.minAreaRect(contours[i])
        points = cv.boxPoints(min_rect).astype(np.int64)
        img2 = cv.drawContours(img2, [points], -1, (0, 255, 0), 2, 8)

    cv.imshow('Max Rect', img1)
    cv.imshow('Min Rect', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
