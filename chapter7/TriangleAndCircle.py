# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 生成空白图像
    image = np.zeros((500, 500))

    # 生成随机点
    points = np.random.randint(150, 270, [100, 2]).astype('float32')

    # 在图像上绘制随机点
    for pt in points:
        cv.circle(image, (pt[0], pt[1]), 1, (255, 255, 255), -1)
    image1 = image.copy()

    # 寻找包围点集的三角形
    _, triangle = cv.minEnclosingTriangle(np.array([points]))
    # 寻找包围点集的圆形
    center, radius = cv.minEnclosingCircle(points)

    # 绘制三角形（为便于读者理解，此处写出了triangle的详细拆分及绘制方式）
    a = triangle[0][0]
    b = triangle[1][0]
    c = triangle[2][0]
    cv.line(image, (a[0], a[1]), (b[0], b[1]), (255, 255, 255), 1, 16)
    cv.line(image, (a[0], a[1]), (c[0], c[1]), (255, 255, 255), 1, 16)
    cv.line(image, (b[0], b[1]), (c[0], c[1]), (255, 255, 255), 1, 16)

    # 绘制圆形
    center = np.int0(center)
    cv.circle(image1, (center[0], center[1]), int(radius), (255, 255, 255), 1, cv.LINE_AA)

    # 展示结果
    cv.imshow('Triangle', image)
    cv.imshow('Circle', image1)
    cv.waitKey(0)
    cv.destroyAllWindows()
