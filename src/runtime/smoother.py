"""smoother.py - 对逐帧概率做 EMA 平滑 + 阈值 + 滞回，输出稳定情绪类别"""

import numpy as np


class EmotionSmoother:
    """EMA 平滑概率向量；切换类别需超过 threshold(+hysteresis) 以抑制抖动。

    锁定类别的平滑概率跌破 threshold-hysteresis 时自动释放，避免卡死在旧标签；
    连续 miss_reset 次无脸（经 note_no_face 上报）后整体重置，避免旧人状态污染新人。
    """

    def __init__(self, num_classes, alpha=0.6, threshold=0.5, hysteresis=0.1, miss_reset=15):
        self.num_classes = num_classes
        self.alpha = alpha
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.miss_reset = miss_reset
        self._ema = None
        self._current = None
        self._misses = 0

    def update(self, probs):
        """输入概率向量，返回 (稳定类别索引或 None, 平滑后概率向量副本)"""
        self._misses = 0
        probs = np.asarray(probs, dtype=np.float32)
        self._ema = probs.copy() if self._ema is None else \
            self.alpha * self._ema + (1 - self.alpha) * probs

        if self._current is not None and \
                float(self._ema[self._current]) < self.threshold - self.hysteresis:
            self._current = None

        top = int(np.argmax(self._ema))
        top_p = float(self._ema[top])
        if self._current is None:
            if top_p >= self.threshold:
                self._current = top
        elif top != self._current and top_p >= self.threshold + self.hysteresis:
            self._current = top
        return self._current, self._ema.copy()

    def note_no_face(self):
        """本帧未检出人脸；连续达到 miss_reset 次则重置全部状态"""
        self._misses += 1
        if self._misses >= self.miss_reset:
            self.reset()

    def reset(self):
        self._ema = None
        self._current = None
        self._misses = 0
