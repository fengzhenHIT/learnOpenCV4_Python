# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像
    image = cv.imread('./images/lena.jpg')
    if image is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 创建SURF对象
    surf = cv.xfeatures2d.SURF_create(500, 4, 3, True, False)

    # 计算SURF特征点
    kps = surf.detect(image, None)

    # 计算SURF描述子
    descriptions = surf.compute(image, kps)

    # 绘制SURF特征点
    image1 = image.copy()
    # 不含角度和大小
    image = cv.drawKeypoints(image, kps, image, ())
    # 包含角度和大小
    image1 = cv.drawKeypoints(image1, kps, image1, (), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 展示结果
    cv.imshow('SURF KeyPoints', image)
    cv.imshow('SURF KeyPoints(with Angle and Size)', image1)
    cv.waitKey(0)
    cv.destroyAllWindows()
