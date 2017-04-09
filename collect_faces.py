import cv2
import json
import numpy as np

from tfyolo import YOLONet

if __name__ == '__main__':
    net = YOLONet('yolo-face.cfg', 'yolo-face.weights')
    faces = []
    metadata = []

    print 'Start capturing video'
    cap = cv2.VideoCapture('video.mp4')
    try:
        idx = 0
        while True:
            result = None
            imgs = []
            for i in range(64):
                res, img = cap.read()
                imgs.append(img)
            imgs = np.stack(imgs)
            result = net.test(imgs)
            if result:
                for cls, x, y, w, h, prob in result:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    #cv2.rectangle(img, (x-w/2, y-h/2), (x+w/2, y+h/2), (255,255,255), 3)
                    roi = img[y-h/2:y+h/2, x-w/2:x+w/2]
                    print "(%05s/71505) (%s, %s) (%s, %s) %s" % (idx, x, y, w, h, prob)
                    if roi.size == 0:
                        continue
                    roi = cv2.resize(roi, (224, 224))
                    roi = roi.reshape((1, 224, 224, 3))
                    faces.append(roi)
                    metadata.append([cls, idx, x, y, w, h, prob])
            cv2.imshow('image', img)
            cv2.waitKey(1)
            if idx % 10 == 0:
                print "(%d/71505)" % (idx)
            idx += 1
    finally:
        np.save('faces.npy', faces)
        with open('metadata.json', 'w') as meta_fp:
            json.dump(metadata, meta_fp)
