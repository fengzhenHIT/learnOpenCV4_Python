# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    src = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25]], dtype='float32')
    dst = cv.flip(src, -1)
    print('原卷积模板为：\n{}'.format(src))
    print('旋转180°后的卷积模板为：\n{}'.format(dst))
