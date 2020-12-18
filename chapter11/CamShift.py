# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    capture = cv.VideoCapture('./data/mulballs.mp4')

    # 判断是否成功加载视频文件
    if not capture.isOpened():
        print('Failed to read mulballs.mp4.')
        sys.exit()

    # 选择跟踪区域
    _, frame = capture.read()
    x, y, w, h = cv.selectROI('CamShift Demo', frame, True, False)
    track_window = (x, y, w, h)

    # 获取ROI直方图
    roi = frame[y: y+h, x: x+w]
    # 将图像转化为HSV颜色空间
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # 阈值操作
    mask = cv.inRange(hsv_roi, 0, 255)
    # 计算直方图和直方图归一化
    roi_hist = cv.calcHist([hsv_roi], [0], hsv_roi, [180], [0, 180])
    roi_hist = cv.normalize(roi_hist, None, 0, 255, cv.NORM_MINMAX)

    # 设置迭代算法终止条件
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = capture.read()
        if frame is None:
            pass
        else:
            frame1 = frame.copy()
        # 当所有帧读取完毕后退出循环
        if ret is False:
            break
        else:
            obj_hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            obj_hist = cv.calcBackProject([obj_hsv], [0], roi_hist, [0, 180], 1)
            # 自适应均值迁移，搜索更新roi区域
            ret, track_window = cv.CamShift(obj_hist, track_window, criteria)

            # 绘制跟踪结果
            x, y, w, h = track_window
            # 利用ret中的信息绘制椭圆形
            # cv.ellipse(frame, ret, (0, 0, 255), thickness=2)
            # 利用track_window中的信息绘制矩形
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 均值迁移，搜索更新roi区域
            ret, track_window = cv.meanShift(obj_hist, track_window, criteria)
            # 绘制跟踪结果
            x, y, w, h = track_window
            cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.imshow('CamShift Demo', frame)
            cv.imshow('MeanShift Demo', frame1)

            # 设置延迟50毫秒，按ESC键退出
            if cv.waitKey(50) & 0xff == 27:
                break

    # 释放并关闭窗口
    capture.release()
    cv.destroyAllWindows()
