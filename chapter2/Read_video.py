# -*- coding:utf-8 -*-
import cv2 as cv


if __name__ == '__main__':
    video = cv.VideoCapture('./videos/road.mp4')

    # 判断是否成功创建视频流
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            cv.imshow('Video', frame)

            # 设置视频播放速度
            # 读者可以尝试将该值做更改，并观看视频播放速度的变化
            cv.waitKey(int(1000 / video.get(cv.CAP_PROP_FPS)))
            # 按下q退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 输出相关信息
    print('视频中图像的宽度为：{}'.format(video.get(cv.CAP_PROP_FRAME_WIDTH)))
    print('视频中图像的高度为：{}'.format(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
    print('视频帧率为：{}'.format(video.get(cv.CAP_PROP_FPS)))
    print('视频总帧数为：{}'.format(video.get(cv.CAP_PROP_FRAME_COUNT)))
    # 释放并关闭窗口
    video.release()
    cv.destroyAllWindows()
