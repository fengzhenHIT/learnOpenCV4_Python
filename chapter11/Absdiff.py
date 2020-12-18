# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    capture = cv.VideoCapture('./data/bike.avi')

    # 判断是否成功加载视频文件
    if not capture.isOpened():
        print('Failed to read bike.avi.')
        sys.exit()

    # 输出视频相关信息
    fps = capture.get(cv.CAP_PROP_FPS)
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    num_of_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
    print('视频宽度为：{}\n视频高度为：{}\n视频帧率为：{}\n视频总帧数为：{}'.format(width, height, fps, num_of_frames))

    # 读取视频中第一帧图像作为前一帧图像，并进行灰度化
    _, pre_frame = capture.read()
    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)
    # 对图像进行高斯滤波，减少噪声干扰
    pre_gray = cv.GaussianBlur(pre_gray, (0, 0), 15)

    # 生成形态学操作的矩阵模板
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7), (-1, -1))
    while True:
        ret, frame = capture.read()
        # 当所有帧读取完毕后退出循环
        if ret is False:
            break
        else:
            # 对当前帧进行灰度化
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (0, 0), 15)

            # 计算当前帧与前一帧的差值的绝对值
            res = cv.absdiff(gray, pre_gray)

            # 对结果进行二值化并进行开运算，以减少噪声干扰
            res = cv.threshold(res, 10, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            res = cv.morphologyEx(res[1], cv.MORPH_OPEN, kernel)

            # 显示结果
            cv.imshow('Origin', frame)
            cv.imshow('Result', res)

            # 将当前帧变为前一帧（注释掉代表以第一帧为固定背景）
            pre_gray = gray.copy()

            # 设置延迟50毫秒，按ESC键退出
            if cv.waitKey(50) & 0xFF == 27:
                break

    # 释放并关闭窗口
    capture.release()
    cv.destroyAllWindows()
