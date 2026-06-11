"""fetch_yunet.py - 下载 YuNet 人脸检测 ONNX 权重到 models/face/"""

import os
import urllib.request

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.face.fetch')

# OpenCV Zoo 官方 YuNet 权重
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def fetch(dest=None):
    dest = dest or str(config.MODELS_DIR / 'face' / 'yunet.onnx')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("下载 YuNet: %s", YUNET_URL)
    urllib.request.urlretrieve(YUNET_URL, dest)
    logger.info("已保存: %s", dest)
    return dest


if __name__ == "__main__":
    fetch()
