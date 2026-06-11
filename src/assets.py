"""assets.py - 运行时资产注册表：人脸检测与情绪模型的下载、缓存与元信息"""

import os
import urllib.request
from dataclasses import dataclass, field

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.assets')


@dataclass(frozen=True)
class Asset:
    """可下载资产：url + 本地路径 +（情绪模型附带）标签与输入尺寸"""
    name: str
    url: str
    dest: str
    labels: tuple = field(default=())
    input_size: int = 0


# YuNet 人脸检测（OpenCV Zoo 官方权重）
YUNET = Asset(
    name='yunet',
    url=('https://github.com/opencv/opencv_zoo/raw/main/models/'
         'face_detection_yunet/face_detection_yunet_2023mar.onnx'),
    dest=str(config.MODELS_DIR / 'face' / 'yunet.onnx'),
)

# HSEmotion EfficientNet-B0（AffectNet 8 类，224×224，ImageNet 归一化，RGB/NCHW）
HSEMOTION_ENET_B0_8 = Asset(
    name='enet_b0_8_best_vgaf',
    url=('https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/'
         'models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx'),
    dest=str(config.MODELS_DIR / 'emotion' / 'enet_b0_8_best_vgaf.onnx'),
    labels=('Anger', 'Contempt', 'Disgust', 'Fear',
            'Happiness', 'Neutral', 'Sadness', 'Surprise'),
    input_size=224,
)

ALL_ASSETS = (YUNET, HSEMOTION_ENET_B0_8)

# Demo 缺省使用的情绪模型
DEFAULT_EMOTION = HSEMOTION_ENET_B0_8


def ensure(asset):
    """资产本地不存在则下载，返回本地路径"""
    if os.path.exists(asset.dest):
        return asset.dest
    os.makedirs(os.path.dirname(asset.dest), exist_ok=True)
    logger.info("下载 %s: %s", asset.name, asset.url)
    urllib.request.urlretrieve(asset.url, asset.dest)
    logger.info("已保存: %s", asset.dest)
    return asset.dest


if __name__ == "__main__":
    for a in ALL_ASSETS:
        ensure(a)
    logger.info("全部资产就绪")
