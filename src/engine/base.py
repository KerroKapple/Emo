"""base.py - 情绪推理引擎抽象：统一前处理(BGR→NCHW 归一化)与 softmax"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

import src.config as config


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max())
    return e / e.sum()


class EmotionEngine(ABC):
    """子类实现 _forward(batch)->logits；engine 自带标签与前处理参数，torch-free"""

    def __init__(self, labels, input_size=112,
                 mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD):
        self.labels = list(labels)
        self.input_size = input_size
        self._mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, face_bgr):
        img = cv2.resize(face_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self._mean) / self._std
        return img[None, ...].astype(np.float32)

    @abstractmethod
    def _forward(self, batch):
        ...

    def infer(self, face_bgr):
        logits = self._forward(self.preprocess(face_bgr))[0]
        return softmax(logits)
