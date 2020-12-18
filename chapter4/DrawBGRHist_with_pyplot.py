# -*- coding:utf-8 -*-
import cv2 as cv
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 读取图像
    img = cv.imread('./images/flower.jpg')
    # 判断图像是否读取成功
    if img is None:
        print('Failed to read flower.jpg.')
        sys.exit()
    # 绘制直方图并展示
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist_item = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist_item, color=col)
    cv.imshow('image', img)
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()
