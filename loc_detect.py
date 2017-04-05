import cv2
import tensorflow as tf
import numpy as np

from tfyolo import YOLONet

if __name__ == '__main__':
    with open("vggface16.tfmodel", mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def)

    graph = tf.get_default_graph()

    #fc8 = graph.get_tensor_by_name("import/prob:0")
    images = graph.get_tensor_by_name("import/images:0")
    fc7 = graph.get_tensor_by_name("import/Relu_1:0")
    id_sess = tf.Session()
    id_sess.run(tf.initialize_all_variables())

    net = YOLONet('yolo-face.cfg', 'yolo-face.weights')
    #img = cv2.imread('face.jpg')
    intents = []
    cap = cv2.VideoCapture('video.mp4')
    for i in xrange(100):
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
            intent, = id_sess.run([fc7], feed_dict={images: roi})
            intents.append(intent)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    np.save('intents.npy', intents)
