# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像hand.png
    image = cv.imread('./images/hand.png')
    if image is None:
        print('Failed to read hand.png.')
        sys.exit()

    # 灰度化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv.threshold(gray, 105, 255, cv.THRESH_BINARY)

    # 对图像进行开运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), (-1, -1))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('Open', binary)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 计算并绘制凸包
    for i in contours:
        # 计算
        hull = cv.convexHull(i)
        # 绘制边缘
        image = cv.drawContours(image, [hull], -1, (0, 0, 255), 2, 8)
        # 绘制顶点
        for j in hull:
            cv.circle(image, (j[0][0], j[0][1]), 4, (255, 0, 0), 2, 8, 0)

    # 展示结果
    cv.imshow('ConvexHull', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
