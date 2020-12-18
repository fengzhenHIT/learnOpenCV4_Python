# -*- coding:utf-8 -*-
import cv2 as cv
import sys


def draw_circle(img, values):
    for i in values[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        cv.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)


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

    # 设置参数
    dp = 2                                  # 离散化累加器分辨率与图像分辨率的反比
    min_dist = 20                           # 两个圆心之间的最小距离
    param1 = 100                            # Canny边缘检测的较大阈值
    param2 = 100                            # 累加器阈值
    min_radius = 20                         # 圆形半径的最小值
    max_radius = 100                        # 圆形半径的最大值

    # 检测圆形
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    # 绘制圆形
    draw_circle(image, circles)

    # 展示结果
    cv.imshow('Detect Circle Result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
