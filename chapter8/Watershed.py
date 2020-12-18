# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def generate_random_color():
    return np.random.randint(0, 256, 3)


def fill_color(img1, n, img2):
    h, w = img1.shape[:-1]
    res = np.zeros((h, w, 3), img1.dtype)
    # 生成随机颜色
    random_color = {}
    for c in range(1, n+1):
        random_color[c] = generate_random_color()
    # 填色
    for i in range(h):
        for j in range(w):
            item = img2[i][j]
            if item == -1:
                res[i, j, :] = (255, 255, 255)
            elif item == 0:
                res[i, j, :] = (0, 0, 0)
            else:
                res[i, j, :] = random_color[item]
    return res


if __name__ == '__main__':
    # 读取图像HoughLines.jpg
    image = cv.imread('./images/HoughLines.jpg')
    if image is None:
        print('Failed to read HoughLines.jpg.')
        sys.exit()
    cv.imshow('Origin', image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 高斯模糊便于减少边缘数目
    # gray = cv.GaussianBlur(gray, (5, 5), 10, sigmaY=20)

    # 提取图像边缘并进行闭运算
    mask = cv.Canny(gray, 150, 300)
    k = cv.getStructuringElement(0, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)
    cv.imshow('mask', mask)

    # 计算连通域数目
    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓，用于输入至分水岭算法
    mask_water = np.zeros(mask.shape, dtype='int32')
    for i in range(len(contours)):
        cv.drawContours(mask_water, contours, i, (i + 1), -1, 8, hierarchy)

    # 分水岭算法操作
    result = cv.watershed(image, mask_water)
    # 为不同的分割区域绘制颜色
    result = fill_color(image, len(contours), mask_water)

    # 展示结果
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
