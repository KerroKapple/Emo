"""config.py - 集中配置：路径、类别常量、输入尺寸与训练超参"""

from dataclasses import dataclass
from pathlib import Path

# 项目根目录：本文件位于 <root>/src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
QUARANTINE_DIR = DATA_DIR / 'quarantine'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

CLASSES = ['anger', 'fear', 'happy', 'sad', 'surprise']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

CLASS_NAMES_ZH = {
    'anger': '愤怒',
    'fear': '恐惧',
    'happy': '快乐',
    'sad': '悲伤',
    'surprise': '惊讶',
}

CLASS_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
}

# 输入尺寸：迁移学习 backbone 用 ImageNet 标准 224；自定义 CNN 用原生 48
# 具体某模型用哪个由 model.get_input_size 决定
TRANSFER_INPUT_SIZE = 224
CNN_INPUT_SIZE = 48

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

# Web 上传体积上限（字节）
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass
class TrainConfig:
    """训练超参配置"""
    model_type: str = 'resnet18'
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    seed: int = 42
    num_workers: int = 4
    early_stop_patience: int = 7
    use_amp: bool = True            # 仅在 CUDA 上生效
    use_class_weights: bool = True  # 按类别频次加权交叉熵
    device: str = 'auto'            # 'auto' | 'cuda' | 'cpu'
