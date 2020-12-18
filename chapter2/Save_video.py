# -*- coding:utf-8 -*-
import cv2 as cv


if __name__ == '__main__':
    # 设置编/解码方式
    fourcc = cv.VideoWriter_fourcc(*'DIVX')

    #  采用摄像头获取图像
    video = cv.VideoCapture(0)
    # cv.VideoWriter()第一种构造函数（两种方法效果相同）
    # result = cv.VideoWriter()
    # result.open('./videos/Save_video.avi', fourcc, 20.0, (640, 480))
    # cv.VideoWriter()第二种构造函数
    result = cv.VideoWriter('./videos/Save_video.avi', fourcc, 20.0, (640, 480))

    # 判断是否成功创建视频流
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            # 将每一帧图像进行水平翻转
            frame = cv.flip(frame, 1)

            # 将一帧一帧图像写入视频
            result.write(frame)
            cv.imshow('Video', frame)
            cv.waitKey(25)

            # 键盘按下q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放并关闭窗口
    video.release()
    result.release()
    cv.destroyAllWindows()
