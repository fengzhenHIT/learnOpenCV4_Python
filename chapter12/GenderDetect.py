# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    image = cv.imread('./images/faces.jpg')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read faces.jpg.')
        sys.exit()

    # 读取人脸识别模型
    face_model = './data/opencv_face_detector_uint8.pb'
    face_config = './data/opencv_face_detector.pbtxt'
    faceNet = cv.dnn.readNet(face_model, face_config)

    # 读取性别检测模型
    gender_model = './data/gender_net.caffemodel'
    gender_config = './data/gender_deploy.prototxt'
    genderNet = cv.dnn.readNet(gender_model, gender_config)

    # 对整幅图像进行人脸检测
    blob = cv.dnn.blobFromImage(image, 1.0, size=(300, 300), swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxes = []
    # 计算图像尺寸
    h, w = image.shape[:-1]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 8, 0)

    padding = -5
    genderList = ['Male', 'Female']
    # 对每个人脸区域进行性别检测
    for bbox in bboxes:
        face = image[max(0, bbox[1] - padding): min(bbox[3] + padding, h - 1),
                     max(0, bbox[0] - padding): min(bbox[2] + padding, w - 1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), swapRB=False, crop=False)
        genderNet.setInput(blob)
        gender_res = genderNet.forward()
        gender = genderList[gender_res[0].argmax()]
        label = '{}'.format(gender)
        cv.putText(image, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 8)

    # 展示结果
    cv.imshow('Result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
