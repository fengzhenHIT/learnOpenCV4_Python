# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys
np.set_printoptions(precision=3, suppress=True)


if __name__ == '__main__':
    # 对矩阵进行处理
    a = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]], dtype='float32')
    b = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
    c = cv.dft(b, flags=cv.DFT_INVERSE | cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    d = cv.idft(b, flags=cv.DFT_SCALE)
    print('正变换结果为：\n{}\n逆变换实数结果为：\n{}\n逆变换结果为：\n{}'.format(b, c, d))

    # 读取图像lena.png
    image = cv.imread('./images/lena.png')
    if image is None:
        print('Failed to read lena.png.')
        sys.exit()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (502, 502))

    image_height = gray.shape[0]
    image_width = gray.shape[1]
    # 计算合适的离散傅里叶变换尺寸
    height = cv.getOptimalDFTSize(image_height)
    width = cv.getOptimalDFTSize(image_width)

    # 扩展图像
    top = int((height - image_height) / 2)
    bottom = int(height - image_height - top)
    left = int((width - image_width) / 2)
    right = int(width - image_width - left)
    appropriate = cv.copyMakeBorder(gray, top=top, bottom=bottom, left=left, right=right, borderType=cv.BORDER_CONSTANT)

    # 计算幅值图像
    # 构建离散傅里叶变换输入量
    flo = np.zeros(appropriate.shape, dtype='float32')
    com = np.dstack([appropriate.astype('float32'), flo])
    # 进行离散傅里叶变换
    result = cv.dft(com, cv.DFT_COMPLEX_OUTPUT)
    # 将变换结果转为幅值
    magnitude_res = cv.magnitude(result[:, :, 0], result[:, :, 1])
    # 进行对数缩放
    magnitude_log = np.log(magnitude_res)
    # 将尺寸对应至原图像
    magnitude_res = magnitude_log[top:image_height, left:image_width]
    # 将结果进行归一化
    magnitude_norm = cv.normalize(magnitude_res, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    # 将幅值中心化处理
    magnitude_center = np.fft.fftshift(magnitude_norm)

    # 展示结果
    cv.imshow('Origin', gray)
    cv.imshow('Border Result', appropriate)
    cv.imshow('Magnitude', magnitude_norm)
    cv.imshow('Magnitude (Center)', magnitude_center)
    cv.waitKey(0)
    cv.destroyAllWindows()
