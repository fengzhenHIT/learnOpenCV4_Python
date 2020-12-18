# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    image = cv.imread('./images/matchTemplate.jpg')
    template = cv.imread('./images/match_template.jpg')
    if image is None or template is None:
        print('Failed to read matchTemplate.jpg or match_template.jpg.')
        sys.exit()
    cv.imshow('image', image)
    cv.imshow('template', template)

    # 计算模板图片的高和宽
    h, w = template.shape[:2]

    # 进行图像模式匹配
    result = cv.matchTemplate(image, template, method=cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # 计算图像左上角、右下角坐标并画出匹配位置
    left_top = max_loc
    right_bottom = (left_top[0] + w, left_top[1] + h)
    cv.rectangle(image, left_top, right_bottom, 255, 2)
    cv.imshow('result', image)

    cv.waitKey(0)
    cv.destroyAllWindows()
