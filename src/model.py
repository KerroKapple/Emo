"""model.py - 表情识别模型定义与工厂"""

import torch
import torch.nn as nn
from torchvision import models

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.model')


class EmotionCNN(nn.Module):
    """自定义 CNN：3 个卷积块（64→128→256）+ 自适应池化解耦输入尺寸 + 全连接分类头"""

    def __init__(self, num_classes=5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        # 自适应池化到固定 6x6，使全连接维度不依赖输入尺寸
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 迁移学习注册表：model_type -> (构造函数, 权重枚举, head 类型, 输入尺寸)
# head 类型: 'fc'=替换 .fc; 'classifier_last'=替换 .classifier[1]; 'vgg'=替换 .classifier[6]
_T = config.TRANSFER_INPUT_SIZE
_TRANSFER_REGISTRY = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, 'fc', _T),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT, 'fc', _T),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, 'fc', _T),
    'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT, 'vgg', _T),
    'mobilenet': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, 'classifier_last', _T),
    'efficientnet': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 'classifier_last', _T),
}

ALL_MODEL_TYPES = ('cnn',) + tuple(_TRANSFER_REGISTRY)


class TransferModel(nn.Module):
    """统一迁移学习包装：加载预训练 backbone 并替换分类头为 num_classes"""

    def __init__(self, model_type, num_classes=5, pretrained=True):
        super().__init__()
        builder, weights_enum, head_kind, _ = _TRANSFER_REGISTRY[model_type]
        weights = weights_enum if pretrained else None
        self.backbone = builder(weights=weights)

        if head_kind == 'fc':
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, num_classes),
            )
        elif head_kind == 'classifier_last':
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        elif head_kind == 'vgg':
            in_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"未知 head 类型: {head_kind}")

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type='cnn', num_classes=5, pretrained=True):
    """模型工厂：'cnn' 用自定义 CNN，其余走迁移学习注册表"""
    if model_type == 'cnn':
        return EmotionCNN(num_classes=num_classes)
    if model_type in _TRANSFER_REGISTRY:
        return TransferModel(model_type, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"不支持的模型类型: {model_type}")


def get_input_size(model_type):
    """返回该模型期望的方形输入边长：迁移学习 224，自定义 CNN 48"""
    if model_type == 'cnn':
        return config.CNN_INPUT_SIZE
    if model_type in _TRANSFER_REGISTRY:
        return _TRANSFER_REGISTRY[model_type][3]
    raise ValueError(f"不支持的模型类型: {model_type}")


def count_parameters(model):
    """统计参数量，返回 (总数, 可训练数)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("参数量: 总 %s | 可训练 %s", f"{total:,}", f"{trainable:,}")
    return total, trainable


if __name__ == "__main__":
    for name in ALL_MODEL_TYPES:
        size = get_input_size(name)
        model = get_model(name, num_classes=config.NUM_CLASSES, pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = model(dummy)
        logger.info("%s | 输入 %dx%d | 输出 %s", name, size, size, tuple(out.shape))
        count_parameters(model)
