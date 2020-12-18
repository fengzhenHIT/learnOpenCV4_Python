# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像qrcode.png
    img = cv.imread('./images/qrcode.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to read qrcode.png.')
        sys.exit()

    # 二维码检测和识别
    qr_detect = cv.QRCodeDetector()
    # 对二维码进行检测
    res, points = qr_detect.detect(img)
    if res:
        print('二维码顶点坐标为：\n{}'.format(points))

        # 对二维码进行解码
        ret, straight_qrcode = qr_detect.decode(img, points)
        print('二维码中信息为：\n{}'.format(ret))
        cv.namedWindow('Straight QRcode', cv.WINDOW_NORMAL)
        cv.imshow('Straight QRcode', straight_qrcode)

    # 定位并解码二维码
    ret1, points1, straight_qrcode1 = qr_detect.detectAndDecode(img)
    # 结果和上述相同，此处不再进行展示
    cv.waitKey(0)
    cv.destroyAllWindows()
