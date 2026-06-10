"""config.py - 集中配置：路径、类别常量、超参"""

from dataclasses import dataclass
from pathlib import Path

# 项目根目录：本文件位于 <root>/src/config.py，向上两级即根
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与产物目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
QUARANTINE_DIR = DATA_DIR / 'quarantine'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 类别
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

# 图像预处理常量（ImageNet 归一化）
# 注意：IMAGE_SIZE 在 P1 保持 48 以不改变现有行为；48→224 的修正属于 P2
IMAGE_SIZE = 48
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 支持的图片扩展名
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


@dataclass
class TrainConfig:
    """训练超参配置"""
    model_type: str = 'resnet18'
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    device: str = 'auto'  # 'auto' | 'cuda' | 'cpu'
