# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像approx.png
    image = cv.imread('./images/approx.png')
    if image is None:
        print('Failed to read approx.png.')
        sys.exit()

    # 灰度化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv.threshold(gray, 105, 255, cv.THRESH_BINARY)

    # 对图像进行开运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), (-1, -1))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 计算图像矩
    for i in contours:
        M = cv.moments(i)
        # Hu距计算
        hu = cv.HuMoments(M)
        print(hu)
