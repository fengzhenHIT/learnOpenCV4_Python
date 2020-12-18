import cv2 as cv
import numpy as np
import datetime
import sys


if __name__ == '__main__':
    image = cv.imread('./images/flower.jpg')
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read flower.jpg.')
        sys.exit()

    # 对比通道的分离
    # 使用opencv中的cv.split()函数
    begin1 = datetime.datetime.now()
    for i in range(100000):
        b1, g1, r1 = cv.split(image)
    end1 = datetime.datetime.now()
    print('通道分离(opencv)：{}s'.format((end1 - begin1).total_seconds()))
    # 使用numpy中的切片和索引
    begin2 = datetime.datetime.now()
    for i in range(100000):
        b2 = image[:, :, 0]
        g2 = image[:, :, 1]
        r2 = image[:, :, 2]
    end2 = datetime.datetime.now()
    print('通道分离(numpy)：{}s'.format((end2 - begin2).total_seconds()))

    # 对比BGR图像转为RGB图像
    # 使用opencv中cv.cvtColor()函数
    begin3 = datetime.datetime.now()
    for i in range(100000):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    end3 = datetime.datetime.now()
    print('BGR转RGB(opencv)：{}s'.format((end3 - begin3).total_seconds()))
    # 使用numpy中的切片和索引
    begin4 = datetime.datetime.now()
    for i in range(100000):
        image_rgb = image[:, :, ::-1]
    end4 = datetime.datetime.now()
    print('BGR转RGB(numpy)：{}s'.format((end4 - begin4).total_seconds()))
