# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def generate_random_color():
    return np.random.randint(0, 256, 3)


def fill_color(img1, n, img2):
    h, w = img1.shape
    res = np.zeros((h, w, 3), img1.dtype)
    # 生成随机颜色
    random_color = {}
    for c in range(1, n):
        random_color[c] = generate_random_color()
    # 为不同的连通域填色
    for i in range(h):
        for j in range(w):
            item = img2[i][j]
            if item == 0:
                pass
            else:
                res[i, j, :] = random_color[item]
    return res


def mark(img, n, stat, cent):
    for i in range(1, n):
        # 绘制中心点
        cv.circle(img, (int(cent[i, 0]), int(cent[i, 1])), 2, (0, 255, 0), -1)
        # 绘制矩形边框
        color = list(map(lambda x: int(x), generate_random_color()))
        cv.rectangle(img,
                     (stat[i, 0], stat[i, 1]),
                     (stat[i, 0] + stat[i, 2], stat[i, 1] + stat[i, 3]),
                     color)
        # 标记数字
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img,
                   str(i),
                   (int(cent[i, 0] + 5), int(cent[i, 1] + 5)),
                   font,
                   0.5,
                   (0, 0, 255),
                   1)


if __name__ == '__main__':
    # 生成待腐蚀图像image
    image = np.array([[0, 0, 0, 0, 255, 0],
                     [0, 255, 255, 255, 255, 255],
                     [0, 255, 255, 255, 255, 0],
                     [0, 255, 255, 255, 255, 0],
                     [0, 255, 255, 255, 255, 0],
                     [0, 0, 0, 0, 0, 0]], dtype='uint8')
    # 分别读取黑背景和白背景图
    black = cv.imread('./images/LearnCV_black.png', cv.IMREAD_GRAYSCALE)
    if black is None:
        print('Failed to read LearnCV_black.png.')
        sys.exit()
    white = cv.imread('./images/LearnCV_white.png', cv.IMREAD_GRAYSCALE)
    if white is None:
        print('Failed to read LearnCV_white.png.')
        sys.exit()
    # 读取米粒图像
    rice = cv.imread('./images/rice.png', cv.IMREAD_GRAYSCALE)
    if rice is None:
        print('Failed to read rice.png.')
        sys.exit()

    # 生成两种结构元素：structure1为矩形结构，structure2为十字结构
    structure1 = cv.getStructuringElement(0, (3, 3))
    structure2 = cv.getStructuringElement(1, (3, 3))

    # 对img1进行腐蚀
    erode_image = cv.erode(image, structure2)
    # 分别对黑背景和白背景图像进行矩形结构和十字结构元素腐蚀
    erode_black_1 = cv.erode(black, structure1)
    erode_black_2 = cv.erode(black, structure2)
    erode_white_1 = cv.erode(white, structure1)
    erode_white_2 = cv.erode(white, structure2)

    # 将图像rice转为二值图像
    rice_BW = cv.threshold(rice, 50, 255, cv.THRESH_BINARY)
    # 对图像进行矩形结构元素腐蚀
    erode_riceBW = cv.erode(rice_BW[1], structure1)
    # 统计连通域
    count, dst, stats, centroids = cv.connectedComponentsWithStats(rice_BW[1], ltype=cv.CV_16U)
    erode_count, erode_dst, erode_stats, erode_centroids = \
        cv.connectedComponentsWithStats(erode_riceBW, ltype=cv.CV_16U)
    # 为不同的连通域填色
    erode_rice = rice
    rice = fill_color(rice, count, dst)
    erode_rice = fill_color(erode_rice, erode_count, erode_dst)
    # 绘制外接矩形及中心点，并进行标记
    mark(rice, count, stats, centroids)
    mark(erode_rice, erode_count, erode_stats, erode_centroids)

    # 展示结果
    cv.namedWindow('image', 0)
    cv.namedWindow('image erode', 0)
    cv.imshow('image', image)
    cv.imshow('image erode', erode_image)
    cv.imshow('LearnCV black', black)
    cv.imshow('LearnCV black erode structure1', erode_black_1)
    cv.imshow('LearnCV black erode structure2', erode_black_2)
    cv.imshow('LearnCV white', white)
    cv.imshow('LearnCV white erode structure1', erode_white_1)
    cv.imshow('LearnCV white erode structure2', erode_white_2)
    cv.imshow('Rice Result', rice)
    cv.imshow('Rice Result erode', erode_rice)

    cv.waitKey(0)
    cv.destroyAllWindows()
