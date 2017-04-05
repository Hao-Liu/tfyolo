import cv2
import tensorflow as tf
import numpy as np

from tfyolo import YOLONet

if __name__ == '__main__':
    net = YOLONet('yolo-face.cfg', 'yolo-face.weights')
    faces = []
    cap = cv2.VideoCapture('video.mp4')
    try:
        while True:
            res, img = cap.read()
            result = net.test(img)

            for cls, x, y, w, h, prob in result:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img, (x-w/2, y-h/2), (x+w/2, y+h/2), (255,255,255), 3)
                roi = img[y-h/2:y+h/2, x-w/2:x+w/2]
                print x, y, w, h, roi.shape
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (224, 224))
                roi = roi.reshape((1, 224, 224, 3))
                faces.append(roi)
            cv2.imshow('image', img)
            cv2.waitKey(1)
    finally:
        np.save('faces.npy', faces)
