# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    origin_image = cv.imread('./images/calcBackProject.jpg')
    template_image = cv.imread('./images/calcBackProject_template.jpg')
    if origin_image is None or template_image is None:
        print('Failed to read calcBackProject.jpg or calcBackProject_template.jpg.')
        sys.exit()
    # 分别将其颜色空间从BGR转换到HSV
    origin_hsv = cv.cvtColor(origin_image, cv.COLOR_BGR2HSV)
    template_hsv = cv.cvtColor(template_image, cv.COLOR_BGR2HSV)

    # 计算模板图像的直方图
    template_hist = cv.calcHist([template_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 对模板图像的直方图进行偏移归一化处理
    cv.normalize(template_hist, template_hist, 0, 255, cv.NORM_MINMAX)
    # 计算直方图的反向投影
    result = cv.calcBackProject([origin_hsv], [0, 1], template_hist, [0, 180, 0, 256], 1)

    # 显示图像
    cv.imshow('Origin Image', origin_image)
    cv.imshow('Template Image', template_image)
    cv.imshow('calcBackProject_result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
