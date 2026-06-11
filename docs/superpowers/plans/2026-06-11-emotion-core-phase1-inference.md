# 情绪核心 Phase 1A：推理闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现硬件无关的实时人脸情绪推理闭环：帧 → YuNet 检测裁脸 → ONNX 情绪模型（可插拔引擎）→ 时序平滑 → 情绪事件，可在普通电脑上演示，核心逻辑在 CPU 上全单测覆盖。

**Architecture:** 三个解耦单元——`face`（检测+裁剪）、`engine`（ONNX 推理，torch-free）、`runtime`（平滑+逐帧编排+Demo）。引擎以 ONNX 为输入、自带标签与前处理参数；可后续替换 RKNN/ncnn 后端而不动 runtime。本计划用一个临时导出的小 ONNX 模型做测试夹具，不依赖真实数据/GPU；真实模型（HSEmotion 预训练或自训蒸馏产物）在 Demo 里按路径注入。

**Tech Stack:** Python, OpenCV(`cv2.FaceDetectorYN` YuNet), onnxruntime(CPU), numpy；复用现有 config/logging/EmotionCNN；pytest。

---

## 文件结构

- `src/face/__init__.py` — 包标识
- `src/face/crop.py` — 纯函数 `crop_face`（扩边裁剪+缩放）
- `src/face/detector.py` — `FaceDetector`（YuNet 封装，可注入 detector 便于测试）
- `src/face/fetch_yunet.py` — 下载 YuNet 模型权重的脚本
- `src/engine/__init__.py` — 包标识
- `src/engine/base.py` — `EmotionEngine` 抽象基类 + 前处理 + softmax
- `src/engine/onnx_engine.py` — `OnnxRuntimeEngine`
- `src/runtime/__init__.py` — 包标识
- `src/runtime/smoother.py` — `EmotionSmoother`（EMA + 阈值 + 滞回）
- `src/runtime/pipeline.py` — `EmotionEvent` + `process_frame` 编排
- `src/runtime/demo.py` — 摄像头/图片 Demo 入口（薄壳，手动运行）
- `tests/_fixtures.py` — 导出临时小 ONNX 模型的测试辅助
- `tests/test_crop.py` / `test_smoother.py` / `test_detector.py` / `test_engine.py` / `test_pipeline.py`

---

## Task 1: 加入依赖并确认环境

**Files:**
- Modify: `pyproject.toml`（经 `uv add`，勿手改）, `uv.lock`

- [ ] **Step 1: 加依赖**

Run:
```bash
uv add opencv-python onnxruntime
```
Expected: 解析并安装成功，新增 `opencv-python`、`onnxruntime`。

- [ ] **Step 2: 验证导入**

Run:
```bash
uv run python -c "import cv2, onnxruntime; print('cv2', cv2.__version__, '| ort', onnxruntime.__version__)"
```
Expected: 打印版本号。

> 若 opencv-python / onnxruntime 在 Python 3.14 无 wheel 而安装失败：新建 `.python-version` 写入 `3.12`，`uv sync` 后重试（与既有"尝鲜版依赖"风险一致）。

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: 加入 opencv-python 与 onnxruntime 依赖"
```

---

## Task 2: 人脸裁剪纯函数 `crop_face`

**Files:**
- Create: `src/face/__init__.py`, `src/face/crop.py`
- Test: `tests/test_crop.py`

- [ ] **Step 1: Write the failing test**

`tests/test_crop.py`:
```python
import numpy as np
from src.face.crop import crop_face


def _img(h=200, w=200):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_crop_returns_requested_size():
    out = crop_face(_img(), (50, 50, 100, 100), size=112)
    assert out.shape == (112, 112, 3)


def test_crop_clamps_at_image_border():
    out = crop_face(_img(200, 200), (180, 180, 100, 100), size=64, margin=0.3)
    assert out.shape == (64, 64, 3)


def test_crop_empty_box_returns_none():
    assert crop_face(_img(), (10, 10, 0, 0)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_crop.py -q`
Expected: FAIL（`ModuleNotFoundError: src.face.crop`）

- [ ] **Step 3: Write minimal implementation**

`src/face/__init__.py`: 空文件。

`src/face/crop.py`:
```python
"""crop.py - 人脸框扩边裁剪并缩放到方形"""

import cv2


def crop_face(image, box, size=112, margin=0.2):
    """box=(x,y,w,h)；按 margin 扩边后裁剪并缩放到 size×size（保持 BGR）。无效框返回 None"""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = image.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(img_w, int(x + w + mx)), min(img_h, int(y + h + my))
    face = image[y0:y1, x0:x1]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_crop.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/face/__init__.py src/face/crop.py tests/test_crop.py
git commit -m "feat: 人脸扩边裁剪纯函数 crop_face"
```

---

## Task 3: 时序平滑 `EmotionSmoother`

**Files:**
- Create: `src/runtime/__init__.py`, `src/runtime/smoother.py`
- Test: `tests/test_smoother.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smoother.py`:
```python
import numpy as np
from src.runtime.smoother import EmotionSmoother


def _onehot(i, n=5, p=0.9):
    v = np.full(n, (1 - p) / (n - 1), dtype=np.float32)
    v[i] = p
    return v


def test_stabilizes_to_sustained_class():
    s = EmotionSmoother(num_classes=5, alpha=0.5, threshold=0.5)
    idx = None
    for _ in range(5):
        idx, _ = s.update(_onehot(2))
    assert idx == 2


def test_single_noisy_frame_does_not_flip():
    s = EmotionSmoother(num_classes=5, alpha=0.7, threshold=0.5, hysteresis=0.15)
    for _ in range(6):
        s.update(_onehot(2))
    idx, _ = s.update(_onehot(0))  # 单帧噪声
    assert idx == 2


def test_sustained_change_flips():
    s = EmotionSmoother(num_classes=5, alpha=0.5, threshold=0.5, hysteresis=0.1)
    for _ in range(6):
        s.update(_onehot(2))
    for _ in range(6):
        idx, _ = s.update(_onehot(0))
    assert idx == 0


def test_below_threshold_stays_none():
    s = EmotionSmoother(num_classes=5, threshold=0.8)
    idx, _ = s.update(np.full(5, 0.2, dtype=np.float32))
    assert idx is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoother.py -q`
Expected: FAIL（`ModuleNotFoundError: src.runtime.smoother`）

- [ ] **Step 3: Write minimal implementation**

`src/runtime/__init__.py`: 空文件。

`src/runtime/smoother.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_smoother.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add src/runtime/__init__.py src/runtime/smoother.py tests/test_smoother.py
git commit -m "feat: 情绪时序平滑 EmotionSmoother（EMA+阈值+滞回）"
```

---

## Task 4: ONNX 推理引擎 `EmotionEngine` / `OnnxRuntimeEngine`

**Files:**
- Create: `src/engine/__init__.py`, `src/engine/base.py`, `src/engine/onnx_engine.py`, `tests/_fixtures.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the fixture helper**

`tests/_fixtures.py`:
```python
"""测试夹具：把一个小 torch 模型导出为临时 ONNX，供引擎测试用（避免依赖真实模型）"""

import torch
from src.model import get_model


def export_tiny_onnx(path, num_classes=5, input_size=112):
    model = get_model('cnn', num_classes=num_classes, pretrained=False).eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy, path, input_names=['input'],
                      output_names=['logits'], opset_version=13)
    return path
```

- [ ] **Step 2: Write the failing test**

`tests/test_engine.py`:
```python
import os
import numpy as np
from tests._fixtures import export_tiny_onnx
from src.engine.onnx_engine import OnnxRuntimeEngine

LABELS = ['anger', 'fear', 'happy', 'sad', 'surprise']


def test_engine_infers_probability_vector(tmp_path):
    path = export_tiny_onnx(str(tmp_path / 'tiny.onnx'), num_classes=len(LABELS))
    engine = OnnxRuntimeEngine(path, LABELS, input_size=112)
    face = (np.random.rand(80, 70, 3) * 255).astype(np.uint8)  # 任意尺寸 BGR
    probs = engine.infer(face)
    assert probs.shape == (len(LABELS),)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert engine.labels == LABELS


def test_preprocess_outputs_nchw(tmp_path):
    path = export_tiny_onnx(str(tmp_path / 'tiny.onnx'), num_classes=len(LABELS))
    engine = OnnxRuntimeEngine(path, LABELS, input_size=112)
    batch = engine.preprocess((np.random.rand(50, 50, 3) * 255).astype(np.uint8))
    assert batch.shape == (1, 3, 112, 112)
    assert batch.dtype == np.float32
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py -q`
Expected: FAIL（`ModuleNotFoundError: src.engine.onnx_engine`）

- [ ] **Step 4: Write minimal implementation**

`src/engine/__init__.py`: 空文件。

`src/engine/base.py`:
```python
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
```

`src/engine/onnx_engine.py`:
```python
"""onnx_engine.py - onnxruntime CPU 后端"""

import onnxruntime as ort

from src.engine.base import EmotionEngine


class OnnxRuntimeEngine(EmotionEngine):
    def __init__(self, model_path, labels, providers=('CPUExecutionProvider',), **kw):
        super().__init__(labels, **kw)
        self._sess = ort.InferenceSession(model_path, providers=list(providers))
        self._input = self._sess.get_inputs()[0].name

    def _forward(self, batch):
        return self._sess.run(None, {self._input: batch})[0]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_engine.py -q`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add src/engine tests/test_engine.py tests/_fixtures.py
git commit -m "feat: ONNX 情绪推理引擎(EmotionEngine + OnnxRuntimeEngine)"
```

---

## Task 5: 人脸检测器 `FaceDetector`（YuNet）

**Files:**
- Create: `src/face/detector.py`, `src/face/fetch_yunet.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing test**

`tests/test_detector.py`:
```python
import numpy as np
from src.face.detector import FaceDetector


class _FakeYuNet:
    """模拟 cv2.FaceDetectorYN：返回一行检测 [x,y,w,h, 10 landmarks, score]"""

    def setInputSize(self, size):
        self._size = size

    def detect(self, image):
        row = np.array([30, 40, 60, 60] + [0] * 10 + [0.9], dtype=np.float32)
        return 1, row[None, :]


def test_detect_parses_boxes():
    det = FaceDetector(detector=_FakeYuNet())
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    boxes = det.detect(img)
    assert boxes == [(30, 40, 60, 60)]


def test_detect_and_crop_returns_face():
    det = FaceDetector(detector=_FakeYuNet())
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    out = det.detect_and_crop(img, size=112)
    assert len(out) == 1
    box, face = out[0]
    assert box == (30, 40, 60, 60)
    assert face.shape == (112, 112, 3)


def test_no_face_returns_empty():
    class _Empty(_FakeYuNet):
        def detect(self, image):
            return 0, None
    det = FaceDetector(detector=_Empty())
    assert det.detect((np.random.rand(50, 50, 3) * 255).astype(np.uint8)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detector.py -q`
Expected: FAIL（`ModuleNotFoundError: src.face.detector`）

- [ ] **Step 3: Write minimal implementation**

`src/face/detector.py`:
```python
"""detector.py - YuNet 人脸检测封装（可注入 detector 便于测试）"""

import cv2

from src.face.crop import crop_face


class FaceDetector:
    """检测人脸框并裁剪。传 model_path 时用 YuNet；传 detector 时直接用（测试用）"""

    def __init__(self, model_path=None, *, detector=None,
                 score_threshold=0.7, nms_threshold=0.3, input_size=(320, 320)):
        if detector is not None:
            self._det = detector
        elif model_path is not None:
            self._det = cv2.FaceDetectorYN.create(
                model_path, "", input_size, score_threshold, nms_threshold)
        else:
            raise ValueError("需提供 model_path 或 detector")

    def detect(self, image):
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        if faces is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in faces]

    def detect_and_crop(self, image, size=112):
        return [(box, crop_face(image, box, size)) for box in self.detect(image)]
```

`src/face/fetch_yunet.py`:
```python
"""fetch_yunet.py - 下载 YuNet 人脸检测 ONNX 权重到 models/face/"""

import urllib.request

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.face.fetch')

# OpenCV Zoo 官方 YuNet 权重
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def fetch(dest=None):
    dest = dest or str(config.MODELS_DIR / 'face' / 'yunet.onnx')
    import os
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("下载 YuNet: %s", YUNET_URL)
    urllib.request.urlretrieve(YUNET_URL, dest)
    logger.info("已保存: %s", dest)
    return dest


if __name__ == "__main__":
    fetch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_detector.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/face/detector.py src/face/fetch_yunet.py tests/test_detector.py
git commit -m "feat: YuNet 人脸检测封装 FaceDetector + 权重下载脚本"
```

---

## Task 6: 逐帧编排 `process_frame` + `EmotionEvent`

**Files:**
- Create: `src/runtime/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import numpy as np
from src.runtime.pipeline import process_frame, EmotionEvent
from src.runtime.smoother import EmotionSmoother

LABELS = ['anger', 'fear', 'happy', 'sad', 'surprise']


class _FakeDetector:
    def __init__(self, crops):
        self._crops = crops

    def detect_and_crop(self, image, size=112):
        return self._crops


class _FakeEngine:
    labels = LABELS
    input_size = 112

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def infer(self, face):
        return self._probs


def test_process_frame_emits_event():
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    det = _FakeDetector([((10, 10, 80, 80), face)])
    eng = _FakeEngine([0.02, 0.02, 0.9, 0.03, 0.03])
    sm = EmotionSmoother(num_classes=5, alpha=0.0, threshold=0.5)  # alpha=0 直接取当前帧
    ev = process_frame(np.zeros((200, 200, 3), np.uint8), det, eng, sm)
    assert isinstance(ev, EmotionEvent)
    assert ev.label == 'happy'
    assert ev.box == (10, 10, 80, 80)


def test_process_frame_no_face_returns_none():
    det = _FakeDetector([])
    eng = _FakeEngine([0.2] * 5)
    sm = EmotionSmoother(num_classes=5)
    assert process_frame(np.zeros((10, 10, 3), np.uint8), det, eng, sm) is None


def test_process_frame_picks_largest_face():
    small = np.zeros((112, 112, 3), np.uint8)
    big = np.zeros((112, 112, 3), np.uint8)
    det = _FakeDetector([((0, 0, 20, 20), small), ((0, 0, 100, 100), big)])
    eng = _FakeEngine([0.02, 0.02, 0.9, 0.03, 0.03])
    sm = EmotionSmoother(num_classes=5, alpha=0.0, threshold=0.5)
    ev = process_frame(np.zeros((200, 200, 3), np.uint8), det, eng, sm)
    assert ev.box == (0, 0, 100, 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL（`ModuleNotFoundError: src.runtime.pipeline`）

- [ ] **Step 3: Write minimal implementation**

`src/runtime/pipeline.py`:
```python
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
        return None

    box, face = max(crops, key=lambda bc: bc[0][2] * bc[0][3])
    probs = engine.infer(face)
    idx, ema = smoother.update(probs)
    if idx is None:
        return None
    return EmotionEvent(label=engine.labels[idx], score=float(ema[idx]), box=box, probs=ema)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add src/runtime/pipeline.py tests/test_pipeline.py
git commit -m "feat: 逐帧编排 process_frame + EmotionEvent"
```

---

## Task 7: Demo 入口 `demo.py`

**Files:**
- Create: `src/runtime/demo.py`
- Test: 复用 `tests/test_pipeline.py` 已覆盖核心逻辑；本任务仅薄壳 + 一个图片冒烟

- [ ] **Step 1: Write the implementation**

`src/runtime/demo.py`:
```python
"""demo.py - 情绪核心 PC 演示：摄像头或图片 → 实时情绪事件

用法:
  uv run python -m src.runtime.demo --camera 0 \
      --face-model models/face/yunet.onnx --emotion-model models/emotion_int8.onnx
  uv run python -m src.runtime.demo --image path/to/face.jpg \
      --face-model models/face/yunet.onnx --emotion-model models/emotion.onnx
"""

import argparse

import cv2

import src.config as config
from src.logging_setup import get_logger
from src.face.detector import FaceDetector
from src.engine.onnx_engine import OnnxRuntimeEngine
from src.runtime.smoother import EmotionSmoother
from src.runtime.pipeline import process_frame

logger = get_logger('emotion.demo')


def build(face_model, emotion_model, labels, input_size):
    detector = FaceDetector(face_model)
    engine = OnnxRuntimeEngine(emotion_model, labels, input_size=input_size)
    smoother = EmotionSmoother(num_classes=len(labels))
    return detector, engine, smoother


def run_image(path, detector, engine, smoother):
    frame = cv2.imread(path)
    if frame is None:
        logger.error("无法读取图片: %s", path)
        return
    event = process_frame(frame, detector, engine, smoother)
    logger.info("情绪: %s", event)


def run_camera(source, detector, engine, smoother):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("无法打开摄像头: %s", source)
        return
    logger.info("按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        event = process_frame(frame, detector, engine, smoother)
        if event:
            x, y, w, h = event.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f"{event.label} {event.score:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.imshow('emotion-core', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def _parse_args():
    p = argparse.ArgumentParser(description='情绪核心 PC 演示')
    p.add_argument('--face-model', required=True)
    p.add_argument('--emotion-model', required=True)
    p.add_argument('--camera', type=int)
    p.add_argument('--image')
    p.add_argument('--input-size', type=int, default=112)
    p.add_argument('--labels', nargs='+', default=config.CLASSES)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    det, eng, sm = build(args.face_model, args.emotion_model, args.labels, args.input_size)
    if args.image:
        run_image(args.image, det, eng, sm)
    else:
        run_camera(args.camera if args.camera is not None else 0, det, eng, sm)
```

- [ ] **Step 2: 冒烟——可导入且 argparse 正常**

Run: `uv run python -m src.runtime.demo --help`
Expected: 打印用法，退出码 0。

- [ ] **Step 3: 全量测试 + lint**

Run:
```bash
uv run pytest -q
uv run ruff check src tests
```
Expected: 全部通过（既有 41 + 新增约 12 用例），ruff 无报错。

- [ ] **Step 4: Commit**

```bash
git add src/runtime/demo.py
git commit -m "feat: 情绪核心 PC 演示入口(摄像头/图片)"
```

---

## Task 8: README 增补「情绪核心」章节

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 增补使用说明**

在 README 适当位置加入：
```markdown
## 🤖 情绪核心（边缘推理）

硬件无关的实时人脸情绪推理闭环，可作陪伴机器人的感知核心。

```bash
# 下载 YuNet 人脸检测权重
uv run python -m src.face.fetch_yunet
# 用已有的情绪 ONNX 模型跑摄像头演示
uv run python -m src.runtime.demo --camera 0 \
    --face-model models/face/yunet.onnx \
    --emotion-model models/emotion.onnx --labels anger fear happy sad surprise
```

链路：取帧 → YuNet 检测裁脸 → ONNX 情绪模型（onnxruntime）→ 时序平滑 → 情绪事件。
后端可替换为 RKNN/ncnn（Phase 2）而不改动 runtime。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README 增补情绪核心用法"
```

---

## 后续计划（不在本计划内）

- **Plan 1B（训练/导出管线，需 GPU+数据）**：`src/data/prepare.py`（YuNet 批量裁脸）、模型注册表加 mobilenetv3/effnet-lite、蒸馏（复用 optimize_distill）、`src/export/to_onnx.py`、`src/export/quantize.py`（INT8 PTQ/QAT 标定），产出自训 `emotion_int8.onnx`。
- **Phase 2（硬件适配，需锁定开发板）**：`RknnEngine` / `NcnnEngine` 后端、板上 FPS/功耗实测、QAT 精度恢复、与机器人屏幕/语音事件对接。
