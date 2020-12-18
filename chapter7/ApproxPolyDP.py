# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np


def judge_shape(val):
    if val == 3:
        return 'Triangle'
    elif val == 4:
        return 'Rectangle'
    else:
        return 'Ploygon-{}'.format(val)


if __name__ == '__main__':
    # 读取图像approx.png
    image = cv.imread('./images/approx.png')
    if image is None:
        print('Failed to read approx.png.')
        sys.exit()

    # 提取图像边缘
    canny = cv.Canny(image, 80, 160, 3)

    # 膨胀运算
    kernel = cv.getStructuringElement(0, (3, 3))
    canny = cv.dilate(canny, kernel=kernel)

    # 轮廓检测及绘制
    contours, hierarchy = cv.findContours(canny, mode=0, method=2)

    for i in range(len(contours)):
        # 多边形拟合
        approx = cv.approxPolyDP(contours[i], 4, closed=True)
        # 多边形绘制
        image = cv.drawContours(image, [approx], -1, (0, 255, 0), 2, 8)
        # 在图中输出多边形形状
        # 计算并绘制多边形形状中心
        center = np.int0((sum(approx)[0] / len(approx)))
        center = (center[0], center[1])
        cv.circle(image, center, 3, (0, 0, 255), -1)
        # 判断并绘制形状信息
        cv.putText(image, text=judge_shape(approx.shape[0]), org=center, fontFace=1, fontScale=1, color=(0, 0, 255))
    cv.imshow('ApproxPolyDP', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
