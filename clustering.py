#!/usr/bin/env python

import cv2
import pyglet
import scipy.ndimage
import numpy as np
from sklearn.manifold import TSNE

class Window(pyglet.window.Window):
    def __init__(self, vertices, faces):
        super(Window, self).__init__()
        label = pyglet.text.Label(
            'Hello, world',
             font_name='Times New Roman',
             font_size=36,
             x=self.width//2, y=self.height//2,
             anchor_x='center', anchor_y='center')
        scale = min(self.width / 2, self.height / 2) / np.max(np.abs(vertices))
        vertices *= scale

        vertices[:, 0] += self.width // 2
        vertices[:, 1] += self.height // 2

        n_vtx = len(vertices)
        self.colors = np.tile((255, 255, 255), n_vtx)
        self.mouse = (0, 0)

        self.vertices_tree = scipy.spatial.cKDTree(vertices, leafsize=100)

        self.vertex_list = pyglet.graphics.vertex_list(
            n_vtx,
            ('v2f', vertices.flatten()), ('c3B', self.colors))

        self.img = self.update_thumbnails(0, 0)

    def update_thumbnails(self, x, y):
        _, indices = self.vertices_tree.query((x, y), k=5)
        self.vertex_list.colors = self.colors
        if isinstance(indices, int):
            indices = [indices]
        for idx in indices:
            self.vertex_list.colors[idx*3: idx*3 + 3] = [255, 0, 0]
        imgs = []
        for idx in indices:
            imgs.append(cv2.resize(faces[idx].astype('uint8')[::-1,::-1,::-1], (24, 24)))
        np_img = np.stack(imgs, 1)
        return pyglet.image.ImageData(24*5, 24, 'RGB', np_img.tostring(), pitch=None)

    def on_draw(self):
        window.clear()
        self.vertex_list.draw(pyglet.gl.GL_POINTS)
        self.img.blit(self.mouse[0], self.mouse[1])

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)
        self.img = self.update_thumbnails(x, y)


if __name__ == '__main__':
    indices = np.random.choice(10000, 1000)
    print 'Loading intents'
    intents = np.load('intents_10000.npy')[indices]
    print 'Loading faces'
    faces = np.load('faces_10000.npy')[indices]
    n_faces, _, width, height, channels = faces.shape
    faces = faces.reshape((n_faces, width, height, channels))
    n_intents, _, intent_size = intents.shape
    intents = np.reshape(intents, (-1, intent_size))
    print 'Building model'
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    print 'Fitting'
    vertices = model.fit_transform(intents)
    window = Window(vertices, faces)
    pyglet.app.run()
