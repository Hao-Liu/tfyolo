import cv2
import numpy as np
import pylab as plt

faces = np.load('faces.npy')
for face in faces:
    cv2.imshow('img', face[0])
    cv2.waitKey(1)


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
