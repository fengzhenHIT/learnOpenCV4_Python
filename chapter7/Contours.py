# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像circles.png
    image = cv.imread('./images/circles.png')
    if image is None:
        print('Failed to read circles.png.')
        sys.exit()
    cv.imshow('Origin', image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 高斯滤波
    gray = cv.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)

    # 二值化
    _, binary = cv.threshold(gray, 75, 180, cv.THRESH_BINARY)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 轮廓绘制
    image = cv.drawContours(image, contours, -1, (0, 0, 255), 2, 8)

    # 输出轮廓结构关系
    print(hierarchy)

    # 展示结果
    cv.imshow('Find and Draw Contours', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
