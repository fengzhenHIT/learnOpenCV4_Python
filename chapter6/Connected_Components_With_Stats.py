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
    # 对图像进行读取，并转换为灰度图像
    rice = cv.imread('./images/rice.png', cv.IMREAD_GRAYSCALE)
    if rice is None:
        print('Failed to read rice.png.')
        sys.exit()

    # 将图像转成二值图像
    rice_BW = cv.threshold(rice, 50, 255, cv.THRESH_BINARY)
    # 统计连通域
    count, dst, stats, centroids = cv.connectedComponentsWithStats(rice_BW[1], ltype=cv.CV_16U)
    # 为不同的连通域填色
    result = fill_color(rice, count, dst)

    # 绘制外接矩形及中心点，并进行标记
    mark(result, count, stats, centroids)

    # 输出每个连通域的面积
    for s in range(1, count):
        print('第 {} 个连通域的面积为：{}'.format(s, stats[s, 4]))

    # 展示结果
    cv.imshow('Origin', rice)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
