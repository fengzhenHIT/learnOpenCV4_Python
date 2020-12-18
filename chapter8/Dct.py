# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys
np.set_printoptions(suppress=True)


if __name__ == '__main__':
    # 对矩阵进行处理
    a = np.array([[1, 2, 3, 4, 5],
                  [2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9]], dtype='float32')
    b = cv.dct(a)
    c = cv.idct(b)
    print('原始数据为：\n{}\nDCT变换后数据为：\n{}\nDCT变换后逆变换结果为：\n{}'.format(a, b, np.int0(c)))

    # 对图像进行处理
    # 读取图像lena.png
    image = cv.imread('./images/lena.png')
    if image is None:
        print('Failed to read lena.png.')
        sys.exit()
    cv.imshow('Origin', image)

    image_height, image_width = image.shape[:-1]
    # 计算合适的离散傅里叶变换尺寸
    height = 2 * cv.getOptimalDFTSize(int((image_height + 1) / 2))
    width = 2 * cv.getOptimalDFTSize(int((image_width + 1) / 2))

    # 扩展图像
    top = 0
    bottom = int(height - image_height - top)
    left = 0
    right = int(width - image_width - left)
    appropriate = cv.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right, borderType=cv.BORDER_CONSTANT)

    # 三个通道需要分别进行DCT变换
    one, two, three = cv.split(appropriate)
    one_DCT = cv.dct(one.astype('float32'))
    two_DCT = cv.dct(two.astype('float32'))
    three_DCT = cv.dct(three.astype('float32'))

    # 进行通道合并
    result = cv.merge([one_DCT, two_DCT, three_DCT])

    # 保存结果
    cv.imwrite('./results/Dct.png', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
