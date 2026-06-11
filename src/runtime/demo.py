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
