# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    capture = cv.VideoCapture('./data/vtest.avi')

    # 判断是否成功加载视频文件
    if not capture.isOpened():
        print('Failed to read vtest.avi.')
        sys.exit()

    # 读取并处理第一帧图像作为函数使用的前一帧图像
    _, pre_frame = capture.read()
    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    # 初始化HSV图像
    hsv = np.zeros_like(pre_frame)
    hsv[..., 1] = 255

    while True:
        _, next_frame = capture.read()
        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        # 计算稠密光流
        flow = cv.calcOpticalFlowFarneback(pre_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 计算向量角度和幅值
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # 将角度转换成角度制
        hsv[..., 0] = angle * 180 / np.pi / 2
        # 将幅值归一化到0-255区间便于显示结果
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # 将HSV颜色空间图像转换到RGB颜色空间中
        result = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # 展示原始图像和结果
        cv.imshow('Origin', next_frame)
        cv.imshow('Object Detect Result', result)

        # 设置延迟50毫秒，按ESC键退出
        if cv.waitKey(50) & 0xff == 27:
            break

    # 释放并关闭窗口
    capture.release()
    cv.destroyAllWindows()
