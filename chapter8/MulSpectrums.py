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
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_float = gray.astype('float32')
    h, w = image.shape[:-1]

    # 构建卷积核
    kernel_w = 5
    kernel_h = 5
    kernel = np.ones((kernel_w, kernel_h), dtype='float32')

    # 计算最优离散傅里叶变换尺寸
    width = cv.getOptimalDFTSize(w + kernel_w - 1)
    height = cv.getOptimalDFTSize(h + kernel_h - 1)

    # 改变输入图像尺寸
    img_tmp = cv.copyMakeBorder(gray_float, 0, height - h, 0, width - w, cv.BORDER_CONSTANT)
    # 改变滤波器尺寸
    kernel_tmp = cv.copyMakeBorder(kernel, 0, height - kernel_h, 0, width - kernel_w, cv.BORDER_CONSTANT)

    # 分别对卷积核和图像进行傅里叶变换
    gray_dft = cv.dft(img_tmp, flags=0, nonzeroRows=w)
    kernel_dft = cv.dft(kernel_tmp, flags=0, nonzeroRows=kernel_w)

    # 多个傅里叶变换结果相乘
    result_mul = cv.mulSpectrums(gray_dft, kernel_dft, cv.DFT_COMPLEX_OUTPUT)

    # 对相乘结果逆变换
    result_idft = cv.idft(result_mul, flags=cv.DFT_SCALE, nonzeroRows=width)

    # 对逆变换结果归一化
    result_norm = cv.normalize(result_idft, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    result = result_norm[0: h, 0: w]
    # 展示结果
    cv.imshow('Origin', gray)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
