import threading
import Queue

from tfyolo import YOLONet


if __name__ == '__main__':
    def _detect(net, info):
        while not info['done']:
            if info['img'] is None:
                time.sleep(0.05)
                continue
            result = net.test(info['img'])
            info['detection'].put(result)

    net = YOLONet('yolo-face.cfg', 'yolo-face.weights')
    info = {'img': None, 'done': False, 'detection': Queue.Queue()}
    detect_thread = threading.Thread(target=_detect, args=[net, info])
    detect_thread.start()
    try:
        cap = cv2.VideoCapture(0)
        result = None
        while True:
            #img = cv2.imread('face.jpg')
            res, img = cap.read()
            info['img'] = img
            #result = net.test(img)
            #width, height, _ = img.shape
            try:
                result = info['detection'].get_nowait()
                for detect in result:
                    print detect
            except Queue.Empty:
                pass

            if result:
                for cls, x, y, w, h, prob in result:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(img, (x-w/2, y-h/2), (x+w/2, y+h/2), (255,255,255), 3)
            cv2.imshow('image', img)
            cv2.waitKey(1)
    finally:
        info['done'] = True
        detect_thread.join()
