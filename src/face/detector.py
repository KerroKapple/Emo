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
