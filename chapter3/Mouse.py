# -*- coding:utf-8 -*-
import cv2 as cv
import sys


def draw(event, x, y, flags, param):
    global img, pre_pts

    # 鼠标右键按下
    if event == cv.EVENT_RBUTTONDOWN:
        print('请点击鼠标左键进行轨迹的绘制。')

    # 鼠标左键按下
    if event == cv.EVENT_LBUTTONDOWN:
        pre_pts = (x, y)
        print('轨迹起始坐标为：{}, {}'.format(x, y))

    # 鼠标移动
    if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        pts = (x, y)
        img = cv.line(img, pre_pts, pts, (0, 0, 255), 2, 5,0)
        pre_pts = pts
        cv.imshow('image', img)


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    img1 = img.copy()
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    pre_pts = -1, -1
    cv.imshow('image', img)
    cv.setMouseCallback('image', draw)
    cv.waitKey(0)
    cv.destroyAllWindows()
