# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 生成一个黑色图像用于绘制图形
    img = np.zeros((512, 512, 3), dtype='uint8')
    # 绘制圆形
    # 绘制实心圆
    img = cv.circle(img, (50, 50), 25, (255, 255, 255), -1)
    # 绘制空心圆
    img = cv.circle(img, (100, 50), 20, (255, 255, 255), 4)

    # 绘制直线
    img = cv.line(img, (100, 100), (200, 100), (255, 255, 255), 2, cv.LINE_4, 0)

    # 绘制椭圆
    img = cv.ellipse(img, (300, 255), (100, 70), 0, 0, 270, (255, 255, 255), -1)

    # 用一些点近似一个椭圆
    points = cv.ellipse2Poly((200, 400), (100, 70), 0, 0, 360, 2)
    # 使用直线将上述点显示出来
    for i in range(len(points) - 1):
        img = cv.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]),
                      (255, 0, 0), 2, cv.LINE_4, 0)
    img = cv.line(img, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]),
                  (255, 0, 0), 2, cv.LINE_4, 0)

    # 绘制矩形
    img = cv.rectangle(img, (50, 400), (100, 450), (0, 255, 0), -1)
    img = cv.rectangle(img, (400, 450, 60, 50), (0, 0, 255), 2)

    # 绘制多边形
    pts = np.array([[350, 83], [463, 90], [500, 171], [421, 194], [338, 141]], dtype='int32')
    img = cv.fillPoly(img, [pts], (255, 0, 0), 8)

    # 添加文字
    img = cv.putText(img, 'Learn OpenCV', (150, 70), 2, 1, (0, 255, 0))
    # 展示结果
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
