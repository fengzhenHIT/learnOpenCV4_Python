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

    # 设置操作标志
    connectivity = 4                # 连通邻域方式
    maskVal = 255                   # 掩码图像的数值
    flags = connectivity | maskVal<<8 | cv.FLOODFILL_FIXED_RANGE    # 漫水填充操作方式标志

    # 设置与选中像素点的差值
    loDiff = (20, 20, 20)
    upDiff = (20, 20, 20)

    # 声明掩模矩阵
    mask = np.zeros((h + 2, w + 2), dtype='uint8')

    while True:
        # 随机选定图像中某一像素点
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        pt = (x, y)

        # 彩色图像中填充像素值
        newVal = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        # 漫水填充
        area = cv.floodFill(image, mask, pt, newVal, loDiff, upDiff, flags)
        # 输出像素点和填充的像素数目
        print('像素点x：{}，y：{}，填充像素数目：{}'.format(x, y, area[0]))
        # 展示结果
        cv.imshow('flood fill', image)
        cv.imshow('mask', mask)
        k = cv.waitKey(0)
        if k == 27:
            break

    cv.destroyAllWindows()
