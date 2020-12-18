# -*- coding:utf-8 -*-
import cv2 as cv
import sys


# 构建高斯金字塔
def gauss_image(image):
    # 设置下采样次数
    level = 3
    img = image.copy()
    gauss_images = []
    gauss_images.append(G0)
    cv.imshow('Gauss_0', G0)
    for i in range(level):
        dst = cv.pyrDown(img)
        gauss_images.append(dst)
        cv.imshow('Gauss_{}'.format(i + 1), dst)
        img = dst.copy()
    return gauss_images


# 构建拉普拉斯金字塔
def laplian_image(image):
    gauss_images = gauss_image(image)
    level = len(gauss_images)
    for i in range(level-1, 0, -1):
        expand = cv.pyrUp(gauss_images[i], dstsize=gauss_images[i-1].shape[:2])
        lpls = cv.subtract(gauss_images[i-1], expand)
        cv.imshow('Laplacian_{}'.format(level-i), lpls)
    # 构建最顶层，需要先进行下采样、再进行上采样
    expand = cv.pyrUp(cv.pyrDown(gauss_images[3]), dstsize=gauss_images[3].shape[:2])
    lpls = cv.subtract(gauss_images[3], expand)
    cv.imshow('Laplacian_{}'.format(0), lpls)


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    G0 = cv.imread('./images/lena.jpg')
    if G0 is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    laplian_image(G0)
    cv.waitKey(0)
    cv.destroyAllWindows()
