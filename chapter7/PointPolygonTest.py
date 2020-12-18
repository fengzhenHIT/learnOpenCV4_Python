# -*- coding:utf-8 -*-
import cv2 as cv
import sys


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

    # 创建图像中的一个点A
    point = (300, 100)

    # 判断点A距离各个轮廓的距离
    for i in range(len(contours)):
        dis = cv.pointPolygonTest(contours[i], point, measureDist=True)
        if dis > 0:
            pos = '内部'
        elif dis == 0:
            pos = '边缘上'
        else:
            pos = '外部'
        print('像素点A（300, 100）距离第{}个轮廓的距离为：{}，'
              '其位置位于轮廓{}'.format(i, round(dis, 2), pos))
