"""demo.py - 情绪核心 PC 演示：摄像头或图片 → 实时情绪事件

开箱即用（自动下载 YuNet + HSEmotion 预训练模型）:
  uv run python -m src.runtime.demo --camera 0
  uv run python -m src.runtime.demo --image path/to/face.jpg

自训模型:
  uv run python -m src.runtime.demo --camera 0 \
      --emotion-model models/emotion_int8.onnx --labels anger fear happy sad surprise \
      --input-size 112
"""

import argparse

import cv2

from src import assets
from src.logging_setup import get_logger
from src.face.detector import FaceDetector
from src.engine.onnx_engine import OnnxRuntimeEngine
from src.runtime.smoother import EmotionSmoother
from src.runtime.pipeline import process_frame

logger = get_logger('emotion.demo')


def build(face_model=None, emotion_model=None, labels=None, input_size=None):
    """构建 检测器/引擎/平滑器；未指定时使用资产注册表缺省（自动下载）"""
    face_model = face_model or assets.ensure(assets.YUNET)
    if emotion_model is None:
        if labels is not None or input_size is not None:
            raise ValueError("--labels/--input-size 仅在指定 --emotion-model 时有效")
        emotion_model = assets.ensure(assets.DEFAULT_EMOTION)
        labels = list(assets.DEFAULT_EMOTION.labels)
        input_size = assets.DEFAULT_EMOTION.input_size
    if not labels or not input_size or input_size <= 0:
        raise ValueError("自定义 --emotion-model 时必须同时给 --labels 与 --input-size")

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
    try:
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
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _parse_args():
    p = argparse.ArgumentParser(description='情绪核心 PC 演示')
    p.add_argument('--face-model', help='缺省自动下载 YuNet')
    p.add_argument('--emotion-model', help='缺省自动下载 HSEmotion enet_b0_8')
    p.add_argument('--camera', type=int)
    p.add_argument('--image')
    p.add_argument('--input-size', type=int)
    p.add_argument('--labels', nargs='+')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    det, eng, sm = build(args.face_model, args.emotion_model, args.labels, args.input_size)
    if args.image:
        run_image(args.image, det, eng, sm)
    else:
        run_camera(args.camera if args.camera is not None else 0, det, eng, sm)
