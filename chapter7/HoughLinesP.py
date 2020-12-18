# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def draw_line(img, lines):
    img_copy = img.copy()
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)
    print(img_copy)
    return img_copy


if __name__ == '__main__':
    # 读取图像HoughLines.jpg
    image = cv.imread('./images/HoughLines.jpg')
    if image is None:
        print('Failed to read HoughLines.jpg.')
        sys.exit()
    cv.imshow('Origin', image)

    # 检测图像边缘
    image_edge = cv.Canny(image, 80, 180, 3)
    cv.imshow('Image Edge', image_edge)

    # 设置直线的最小长度
    min_line_length = 200

    # 分别设定不同直线最大连接距离进行直线检测，并显示结果
    max_line_gap_1 = 5
    lines_1 = cv.HoughLinesP(image_edge, 1, np.pi / 180, 150, minLineLength=min_line_length, maxLineGap=max_line_gap_1)

    img1 = draw_line(image, lines_1)
    # try:
    cv.imshow('Image HoughLinesP ({})'.format(max_line_gap_1), img1)
    # except TypeError:
    #     print('最大连接距离设为 {} 时，不能检测出直线.'.format(max_line_gap_1))
    print(lines_1)
    max_line_gap_2 = 20
    lines_2 = cv.HoughLinesP(image_edge, 1, np.pi / 180, 150, minLineLength=min_line_length, maxLineGap=max_line_gap_2)
    try:
        img2 = draw_line(image, lines_2)
        cv.imshow('Image HoughLinesP ({})'.format(max_line_gap_2), img2)
    except TypeError:
        print('最大连接距离设为 {} 时，不能检测出直线.'.format(max_line_gap_2))

    cv.waitKey(0)
    cv.destroyAllWindows()
