# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像lena.png
    image1 = cv.imread('./images/inpaint1.png')
    image2 = cv.imread('./images/inpaint2.png')
    if image1 is None or image2 is None:
        print('Failed to read inpaint1.png or inpaint2.png.')
        sys.exit()
    cv.imshow('Origin1', image1)
    cv.imshow('Origin2', image2)

    # 生成Mask掩模
    _, mask1 = cv.threshold(image1, 245, 255, cv.THRESH_BINARY)
    _, mask2 = cv.threshold(image2, 254, 255, cv.THRESH_BINARY)
    # 对Mask膨胀处理，增加其面积
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask1 = cv.dilate(mask1, k)
    mask2 = cv.dilate(mask2, k)
    cv.imshow('Mask1', mask1)
    cv.imshow('Mask2', mask2)
    # 图像修复
    result1 = cv.inpaint(image1, mask1[:, :, -1], 5, cv.INPAINT_NS)
    result2 = cv.inpaint(image2, mask2[:, :, -1], 5, cv.INPAINT_NS)

    # 展示结果
    cv.imshow('Result1', result1)
    cv.imshow('Result2', result2)
    cv.waitKey(0)
    cv.destroyAllWindows()
