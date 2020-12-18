# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像lena.png
    image = cv.imread('./images/lena.png')
    if image is None:
        print('Failed to read lena.png.')
        sys.exit()
    h, w = image.shape[:-1]
    # 备份图像，防止绘制矩形框对结果产生影响
    imgRect = image.copy()
    imgRect = cv.rectangle(imgRect, (80, 30), (420, 420), (255, 255, 255), 2)
    cv.imshow('Select Area', imgRect)

    # 进行分割
    bgdmod = np.zeros((1, 65), dtype='float64')
    fgdmod = np.zeros((1, 65), dtype='float64')
    mask = np.zeros(image.shape[:-1], dtype='uint8')
    mask, _, _ = cv.grabCut(image, mask, rect=(80, 30, 420, 420), bgdModel=bgdmod, fgdModel=fgdmod,
               iterCount=5, mode=cv.GC_INIT_WITH_RECT)

    # 将分割出的前景绘制出来
    for i in range(h):
        for j in range(w):
            n = mask[i, j]
            if n == 1 or n == 3:
                pass
            else:
                image[i, j, :] = 0

    # 展示结果
    cv.imshow('Result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
