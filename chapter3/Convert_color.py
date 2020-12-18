# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    else:

        # 将图像进行颜色模型转换
        image = img.astype('float32')
        image *= 1.0 / 255
        HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        Lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        GRAY = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # 展示结果
        cv.imshow('Origin Image', image)
        cv.imshow('HSV Image', HSV)
        cv.imshow('YUV Image', YUV)
        cv.imshow('Lab Image', Lab)
        # 由于计算出Lab结果会有负数值，不能通过cv.imshow()函数显示
        # 因此我们可以使用cv.imwrite()函数保存下来进行查看
        cv.imwrite('./results/Convert_color_Lab.jpg', Lab)
        cv.imshow('GRAY Image', GRAY)

        # 关闭窗口
        cv.waitKey(0)
        cv.destroyAllWindows()
