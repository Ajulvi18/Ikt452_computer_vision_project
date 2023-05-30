import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as TF
import tensorflow as tf
class MorphologicalOperations:
    def __init__(self, kernel_size=5):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
        input = cv.convertScaleAbs(input)
        opening = cv.morphologyEx(input, cv.MORPH_OPEN, self.kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, self.kernel)
        return closing


class MorphologicalOperations_:
    def __init__(self):
        pass
    def __call__(self, input, kernel):
        opening = self.opening(input, kernel)
        closing = self.closing(opening, kernel)
        return closing
    def opening(self, input, kernel):
        dilation = tf.nn.dilation2d(input, filters=kernel, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], padding='SAME', name=None, data_format='NHWC')
        erosion = tf.nn.erosion2d(dilation, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
        return erosion

    def closing(self, input, kernel):
        erosion = tf.nn.erosion2d(input, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
        dilation = tf.nn.dilation2d(erosion, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
        return dilation
#https://github.com/lucasb-eyer/pydensecrf
class CRF:
    def __init__(self):
        pass
    def __call__(self, input):
        output = input.detach().numpy()

        return torch.from_numpy(output)

