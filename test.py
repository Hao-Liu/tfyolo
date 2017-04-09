import cv2
import numpy as np
import tensorflow as tf
import pylab as plt

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

    faces = np.load('faces.npy')
    intents = []
    for idx, face in enumerate(faces):
        print "%d/%d" % (idx, len(faces))
        intent, = id_sess.run([fc7], feed_dict={images: face}) intents.append(intent)
        cv2.imshow('img', face[0])
        cv2.waitKey(1)
    np.save('intents.npy', intents)
    exit(0)


intents = np.load('intents.npy')
n = len(intents)
dist_mat = np.zeros((n, n))
for i, intent_i in enumerate(intents):
    for j, intent_j in enumerate(intents):
        dist_mat[i, j] = np.linalg.norm(intent_i - intent_j)
print dist_mat

plt.matshow(dist_mat)
plt.colorbar()
plt.show()
