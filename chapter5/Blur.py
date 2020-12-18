# -*- coding:utf-8 -*-
import cv2 as cv
import sys


def my_blur(image):
    return cv.blur(image, (3, 3)), cv.blur(image, (9, 9))


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/Gray_dolphins.jpg')
    if img is None:
        print('Failed to read Gray_dolphins.jpg.')
        sys.exit()

    img_sp = cv.imread('./images/GraySaltPepperImage.jpg')
    if img_sp is None:
        print('Failed to read GraySaltPepperImage.jpg.')
        sys.exit()

    img_gauss = cv.imread('./images/GrayGaussImage.jpg')
    if img_gauss is None:
        print('Failed to read GrayGaussImage.jpg.')
        sys.exit()

    img1, img2 = my_blur(img)
    img_sp1, img_sp2 = my_blur(img_sp)
    img_gauss1, img_gauss2 = my_blur(img_gauss)

    # 展示结果
    cv.imshow('Origin Image', img)
    cv.imshow('3 * 3 Blur Image', img1)
    cv.imshow('5 * 5 Blur Image', img2)

    cv.imshow('Origin sp-noisy Image', img_sp)
    cv.imshow('3 * 3 sp-noisy Blur Image', img_sp1)
    cv.imshow('5 * 5 sp-noisy Blur Image', img_sp2)

    cv.imshow('Origin gauss-noisy Image', img_gauss)
    cv.imshow('3 * 3 gauss-noisy Blur Image', img_gauss1)
    cv.imshow('5 * 5 gauss-noisy Blur Image', img_gauss2)

    cv.waitKey(0)
    cv.destroyAllWindows()
