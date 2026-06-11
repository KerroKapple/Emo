"""transforms.py - 按模型类型构建图像预处理/增强管线"""

from torchvision import transforms

import src.config as config
from src.model import get_input_size


def build_transform(model_type, train=False):
    """返回与 model_type 输入尺寸匹配的 transform；train=True 时附带数据增强"""
    size = get_input_size(model_type)
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
