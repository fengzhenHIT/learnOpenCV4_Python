# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


# 设定bins的数目
bins = np.arange(256).reshape(256, 1)


def draw_gray_histogram(image):
    # 创建一个全0矩阵以绘制直方图
    new = np.zeros((image.shape[0], 256, 3))
    # 对图像进行直方图计算
    hist_item = cv.calcHist([image], [0], None, [256], [0, 256])
    # 对直方图进行归一化，我们将在4.3.1节进行详细讲解
    cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        cv.line(new, (x, 0), (x, y), (255, 255, 255))
    # 由于绘制时是从顶部开始绘制，因此需要将矩阵进行翻转
    result = cv.flip(new, 0)
    return result


def draw_bgr_histogram(image):
    # 创建一个3通道的全0矩阵以绘制直方图
    new = np.zeros((image.shape[0], 256, 3))
    # 声明BGR三种颜色
    bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, col in enumerate(bgr):
        hist_item = cv.calcHist([image], [i], None, [256], [0, 256])
        cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        hist = np.int32(np.column_stack((bins, hist)))
        cv.polylines(new, [hist], False, col)
    result = cv.flip(new, 0)
    return result


if __name__ == '__main__':
    # 读取图像flower.jpg
    img = cv.imread('./images/flower.jpg')
    # 判断是否读取成功
    if img is None:
        print("Failed to read flower.jpg.")
        sys.exit()
    # 将图片转为灰度模式
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 计算并绘制灰度图像直方图和BGR图像直方图
    gray_histogram = draw_gray_histogram(gray)
    bgr_histogram = draw_bgr_histogram(img)

    cv.imshow('Origin Image', img)
    cv.imshow('Gray Histogram', gray_histogram)
    cv.imshow('BGR Histogram', bgr_histogram)

    cv.waitKey(0)
    cv.destroyAllWindows()
