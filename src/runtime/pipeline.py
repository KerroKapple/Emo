"""pipeline.py - 单帧编排：检测裁脸 → 引擎推理 → 平滑 → 情绪事件"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionEvent:
    label: str
    score: float
    box: tuple
    probs: np.ndarray


def process_frame(frame, detector, engine, smoother):
    """返回稳定情绪事件；无脸或未达阈值返回 None。多脸取面积最大者（最近的人）"""
    crops = detector.detect_and_crop(frame, size=engine.input_size)
    crops = [(box, face) for box, face in crops if face is not None]
    if not crops:
        smoother.note_no_face()
        return None

    box, face = max(crops, key=lambda bc: bc[0][2] * bc[0][3])
    probs = engine.infer(face)
    idx, ema = smoother.update(probs)
    if idx is None:
        return None
    return EmotionEvent(label=engine.labels[idx], score=float(ema[idx]), box=box, probs=ema)
