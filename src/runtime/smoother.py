"""smoother.py - 对逐帧概率做 EMA 平滑 + 阈值 + 滞回，输出稳定情绪类别"""

import numpy as np


class EmotionSmoother:
    """EMA 平滑概率向量；切换类别需超过 threshold(+hysteresis) 以抑制抖动"""

    def __init__(self, num_classes, alpha=0.6, threshold=0.5, hysteresis=0.1):
        self.num_classes = num_classes
        self.alpha = alpha
        self.threshold = threshold
        self.hysteresis = hysteresis
        self._ema = None
        self._current = None

    def update(self, probs):
        """输入概率向量，返回 (稳定类别索引或 None, 平滑后概率向量)"""
        probs = np.asarray(probs, dtype=np.float32)
        self._ema = probs.copy() if self._ema is None else \
            self.alpha * self._ema + (1 - self.alpha) * probs

        top = int(np.argmax(self._ema))
        top_p = float(self._ema[top])
        if self._current is None:
            if top_p >= self.threshold:
                self._current = top
        elif top != self._current and top_p >= self.threshold + self.hysteresis:
            self._current = top
        return self._current, self._ema

    def reset(self):
        self._ema = None
        self._current = None
