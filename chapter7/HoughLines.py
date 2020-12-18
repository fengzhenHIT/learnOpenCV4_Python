# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def draw_line(img, lines):
    img_copy = img.copy()
    for i in range(0, len(lines)):
        rho, theta = lines[i][0][0], lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img_copy


if __name__ == '__main__':
    # 读取图像HoughLines.jpg
    image = cv.imread('./images/HoughLines.jpg')
    if image is None:
        print('Failed to read HoughLines.jpg.')
        sys.exit()
    cv.imshow('Origin', image)

    # 检测图像边缘
    image_edge = cv.Canny(image, 50, 150, 3)
    cv.imshow('Image Edge', image_edge)

    # 分别设定不同累加器阈值进行直线检测，并显示结果
    threshold_1 = 200
    lines_1 = cv.HoughLines(image_edge, 1, np.pi / 180, threshold_1)
    try:
        img1 = draw_line(image, lines_1)
        cv.imshow('Image HoughLines({})'.format(threshold_1), img1)
    except TypeError:
        print('累加器阈值设为 {} 时，不能检测出直线.'.format(threshold_1))

    threshold_2 = 300
    lines_2 = cv.HoughLines(image_edge, 1, np.pi / 180, threshold_2)
    try:
        img2 = draw_line(image, lines_2)
        cv.imshow('Image HoughLines({})'.format(threshold_2), img2)
    except TypeError:
        print('累加器阈值设为 {} 时，不能检测出直线.'.format(threshold_2))

    cv.waitKey(0)
    cv.destroyAllWindows()
