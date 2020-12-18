# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    capture = cv.VideoCapture('./data/mulballs.mp4')
    # 判断是否成功加载视频文件
    if not capture.isOpened():
        print('Failed to read mulballs.mp4.')
        sys.exit()

    # 随机选取颜色
    color = np.random.randint(0, 255, (100, 3))

    # 读取第一帧
    _, pre_frame = capture.read()
    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)
    # 进行角点检测
    points = cv.goodFeaturesToTrack(pre_gray, maxCorners=5000, qualityLevel=0.01, minDistance=10,
                                    blockSize=3, useHarrisDetector=False, k=0.04)

    # 光流跟踪
    while True:
        ret, frame = capture.read()
        if ret is False:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 稀疏光流检测
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
        next_pts, status, err = cv.calcOpticalFlowPyrLK(pre_gray, frame_gray, points, None, winSize=(31, 31),
                                                        maxLevel=3, criteria=criteria, flags=0)

        # 根据状态对角点做筛选
        good_next = next_pts[status == 1]
        good_pre = points[status == 1]

        # 绘制跟踪线
        for i, (next_item, pre_item) in enumerate(zip(good_next, good_pre)):
            a, b = next_item.ravel()
            c, d = pre_item.ravel()
            # 设置阈值，只绘制移动的角点
            dist = abs(a - c) + abs(b - d)
            if dist > 2:
                frame = cv.circle(frame, (a, b), 3, color[i].tolist(), -1, 8)
                frame = cv.line(frame, (a, b), (c, d), color[i].tolist(), 2, 8, 0)

        # 展示结果
        cv.imshow('Result', frame)
        # 设置延迟50毫秒，按ESC键退出
        if cv.waitKey(50) & 0xff == 27:
            break

        # 更新前一帧图像和角点坐标
        pre_gray = frame_gray.copy()
        points = good_next.reshape(-1, 1, 2)

    # 释放并关闭窗口
    cv.destroyAllWindows()
    capture.release()
