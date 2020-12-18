# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像ABC.png
    image1 = cv.imread('./images/ABC.png')
    if image1 is None:
        print('Failed to read ABC.png.')
        sys.exit()

    image2 = cv.imread('./images/B.png')
    if image2 is None:
        print('Failed to read B.png.')
        sys.exit()
    cv.imshow('B', image2)

    # 灰度化
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    # 二值化
    _, binary1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY)
    _, binary2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY)

    # 轮廓检测
    contours1, _ = cv.findContours(binary1, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv.findContours(binary2, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # Hu距计算
    hu = cv.HuMoments(cv.moments(contours2[0]))

    # 轮廓匹配
    for i in range(len(contours1)):
        hu1 = cv.HuMoments(cv.moments(contours1[i]))
        dist = cv.matchShapes(hu1, hu, cv.CONTOURS_MATCH_I1, 0)
        if dist < 1:
            cv.drawContours(image1, contours1, i, (0, 0, 255), 3, 8)

    # 展示结果
    cv.imshow('Match Result', image1)
    cv.waitKey(0)
    cv.destroyAllWindows()
