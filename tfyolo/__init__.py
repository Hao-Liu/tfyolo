#!/usr/bin/env python

import ConfigParser
import collections
import re
import struct
import time

import cv2
import numpy as np
import tensorflow as tf
import pylab as plt


# http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def do_nms(boxes, overlapThresh):
    if len(boxes) == 0:
	return boxes

    pick = []

    w = boxes[:,2]
    h = boxes[:,3]
    x1 = boxes[:,0] - w / 2.
    y1 = boxes[:,1] - h / 2.
    x2 = boxes[:,0] + w / 2.
    y2 = boxes[:,1] + h / 2.

    area = (w + 1) * (h + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
	last = len(idxs) - 1
	i = idxs[last]
	pick.append(i)

	xx1 = np.maximum(x1[i], x1[idxs[:last]])
	yy1 = np.maximum(y1[i], y1[idxs[:last]])
	xx2 = np.minimum(x2[i], x2[idxs[:last]])
	yy2 = np.minimum(y2[i], y2[idxs[:last]])

	w = np.maximum(0, xx2 - xx1 + 1)
	h = np.maximum(0, yy2 - yy1 + 1)

	overlap = (w * h) / area[idxs[:last]]

	idxs = np.delete(
                idxs,
                np.concatenate(
                    ([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]


class YOLONet(object):
    def __init__(self, config_path, weight_path):
        self.classes = ['face']
        self.threshold = 0.2
        self.iou_threshold = 0.5
        with open(config_path) as fp:
            config = fp.read()
        self.parse(config)
        self.load_weights(weight_path)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def parse(self, config):
        current_section = None
        self.sections = []
        for line in config.splitlines():
            line = line.strip()
            if not line:
                continue
            section_match = re.match(r'^\[(.*)\]$', line)
            if section_match:
                if current_section:
                    self.sections.append(current_section)
                current_section = {'type': section_match.group(1)}
                continue
            set_match = re.match(r'^(.*)=(.*)$', line)
            if set_match:
                if current_section is None:
                    raise ValueError(
                            "Config file should starts with section name, "
                            "'%s' found" % line)
                key, value = set_match.groups()
                current_section[key] = value
                continue
        self.sections.append(current_section)

    def test(self, img):
        start = time.time()
        height, width, _ = img.shape
        img = cv2.resize(img, (448, 448))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # TODO: fix by using actual average subtraction
        inputs = np.expand_dims(img, axis=0) / 255.0 * 2.0 - 1.0
        net_output, = self.sess.run(
                [self.net],
                feed_dict={self.inputs: inputs})
        self.result = self.convert_detections(net_output[0], width, height)
        duration = str(time.time() - start)
        print 'Duration: %s' % duration
        return self.result

    def convert_detections(self, output, width, height):
        prob_range = [0, self.grid_size * self.grid_size * self.num_class]
        scales_range = [prob_range[1], prob_range[1] + self.grid_size * self.grid_size * self.num_box]
        boxes_range = [scales_range[1], scales_range[1] + self.grid_size * self.grid_size * self.num_box * 4]

        probs = np.zeros((self.grid_size, self.grid_size, self.num_box, self.num_class))
        class_probs = np.reshape(
                output[0:prob_range[1]],
                (self.grid_size, self.grid_size, self.num_class))
        scales = np.reshape(
                output[scales_range[0]:scales_range[1]],
                (self.grid_size, self.grid_size, self.num_box))
        boxes = np.reshape(
                output[boxes_range[0]:],
                (self.grid_size, self.grid_size, self.num_box, 4))
        offset = np.transpose(
                np.reshape(
                    np.array([np.arange(self.grid_size)] * (2 * self.grid_size)),
                    (2, self.grid_size, self.grid_size)),
                (1, 2, 0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,:2] = boxes[:,:,:,:2] / float(self.grid_size)
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        boxes[:,:,:,0] *= width
        boxes[:,:,:,1] *= height
        boxes[:,:,:,2] *= width
        boxes[:,:,:,3] *= height

        for i in range(self.num_box):
            for j in range(self.num_class):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        filter_iou = np.array(probs_filtered>0.0, dtype='bool')
        boxes_filtered = do_nms(boxes_filtered[filter_iou], 0.5)
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([
                self.classes[classes_num_filtered[i]],
                boxes_filtered[i][0],
                boxes_filtered[i][1],
                boxes_filtered[i][2],
                boxes_filtered[i][3],
                probs_filtered[i]])

        return result

    def load_conv(self, filters, size):
        print 'convolution', filters, self.channels, size, size
        bias = struct.unpack("%df" % filters, self.weights_content[self.pointer:self.pointer+filters*4])
        self.pointer += 4 * filters

        n = filters * self.channels * size * size
        weights = struct.unpack("%df" % n, self.weights_content[self.pointer:self.pointer+n*4])
        self.pointer += n * 4
        # TODO: check order
        weights = np.reshape(
                np.array(weights, np.float32),
                [filters, self.channels, size, size])
        # [size, size, self.channels, filters])
        weights = np.transpose(weights, [2, 3, 1, 0])
        return bias, weights

    def load_conn(self, inputs, output):
        print 'connect', inputs, output
        bias = struct.unpack("%df" % output, self.weights_content[self.pointer:self.pointer+output*4])
        self.pointer += output * 4
        size = inputs * output
        weights = struct.unpack("%df" % size, self.weights_content[self.pointer:self.pointer+size*4])
        self.pointer += size * 4
        # TODO: check order
        weights = np.reshape(
                np.array(weights, np.float32),
                [output, inputs])
        weights = np.transpose(weights, [1, 0])
        return bias, weights

    def load_weights(self, weight_path):
        with open(weight_path, mode='rb') as fp_weight:
            self.weights_content = fp_weight.read()
        self.pointer = 0
        major, minor, revision, seen = struct.unpack("iiii", self.weights_content[:16])
        print major, minor, revision, seen
        self.pointer += 16
        for idx, section in enumerate(self.sections):
            if section['type'] == 'net':
                self.height = int(section['height'])
                self.width = int(section['width'])
                self.channels = int(section['channels'])
                self.inputs = self.net = tf.placeholder(
                        'float32',
                        [None, self.height, self.width, self.channels])
            elif section['type'] == 'convolutional':
                pad = int(section['pad'])
                size = int(section['size'])
                filters = int(section['filters'])
                stride = int(section['stride'])
                change = pad * 2 - size + 1
                if change == 0:
                    padding = 'SAME'
                else:
                    raise Exception("Unhandled padding. (pad:%d size:%d)" %
                            (pad, size))
                bias, weight = self.load_conv(filters, size)
                bias, weight = tf.Variable(bias), tf.Variable(weight)
		conv = tf.nn.conv2d(
                        self.net,
                        weight,
                        strides=[1, stride, stride, 1],
                        padding=padding)
                conv_biased = tf.add(conv, bias)
                # TODO: fix memory issue
                # https://github.com/tensorflow/tensorflow/issues/4079
		self.net = tf.maximum(
                        0.1 * conv_biased,
                        conv_biased, name='conv_%s' % idx)
                # TODO: deal with stride
                self.width = self.width + change
                self.height = self.height + change
                self.channels = int(section['filters'])
            elif section['type'] == 'maxpool':
                size = int(section['size'])
                # TODO: deal with stride
                stride = int(section['stride'])
                self.width /= size
                self.height /= size
		self.net = tf.nn.max_pool(
                        self.net,
                        ksize=[1, size, size, 1],
                        strides=[1, stride, stride, 1],
                        padding='SAME')
            elif section['type'] == 'connected':
                output = int(section['output'])
                activation = section['activation']
                inputs = self.width * self.height * self.channels
                if len(self.net.get_shape()) > 2:
                    self.net = tf.transpose(self.net, [0, 3, 1, 2], name='transposed')
                    self.net = tf.reshape(self.net, [-1, inputs], name='flattened')
                bias, weight = self.load_conn(inputs, output)
                bias, weight = tf.Variable(bias), tf.Variable(weight)
                #self.net = tf.matmul(self.net, weight, name='fc_%s' % idx)
                self.net = tf.add(
                        tf.matmul(self.net, weight),
                        bias, name='fc_%s' % idx)
		if activation == 'linear':
                    pass
                elif activation == 'leaky':
                    self.net = tf.maximum(
                            0.1 * self.net,
                            self.net, name='fc_%s' % idx)
                else:
                    raise Exception("Unhandled activation function: %s",
                            activation)
                self.width = 1
                self.height = output
                self.channels = 1
            elif section['type'] == 'detection':
                self.num_class = int(section['classes'])
                self.num_box = int(section['num'])
                self.grid_size = int(section['side'])
