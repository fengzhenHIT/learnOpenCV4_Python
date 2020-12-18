# -*- coding:utf-8 -*-
import cv2 as cv
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 读取图像road.jpg
    image = cv.imread('./images/road.jpg')
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read image.')
        sys.exit()
    # 将图像的颜色空间从BGR转为HSV
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # 计算2D直方图
    image_hist = cv.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 展示图像及直方图结果
    cv.imshow('Origin Image', image)
    plt.imshow(image_hist, interpolation='nearest')
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()
