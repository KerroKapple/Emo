"""inference.py - 自描述 checkpoint 的保存/加载与单图推理

checkpoint 内自带 model_type/classes/input_size，加载方无需靠文件名猜测模型类型。
"""

import os

import torch
import torch.nn as nn

import src.config as config
from src.model import get_model, get_input_size
from src.logging_setup import get_logger

logger = get_logger('emotion.inference')


def resolve_device(device='auto'):
    """把 'auto'/'cuda'/'cpu' 解析为 torch.device"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def save_checkpoint(model, model_type, path, *, epoch=None, val_loss=None,
                    val_acc=None, optimizer=None, quantized=False, pruned=False, extra=None):
    """保存自描述 checkpoint"""
    payload = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'classes': config.CLASSES,
        'num_classes': config.NUM_CLASSES,
        'input_size': get_input_size(model_type),
        'epoch': epoch,
        'val_loss': val_loss,
        'accuracy': val_acc,
        'quantized': quantized,
        'pruned': pruned,
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if extra:
        payload.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    logger.info("已保存 checkpoint: %s (acc=%s)", path, val_acc)


def load_model_from_checkpoint(path, device='auto', *, eval_mode=True):
    """从自描述 checkpoint 构建并加载模型，返回 (model, info)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    device = resolve_device(device)
    checkpoint = torch.load(path, map_location=device)

    model_type = checkpoint['model_type']
    num_classes = checkpoint.get('num_classes', config.NUM_CLASSES)
    model = get_model(model_type, num_classes=num_classes, pretrained=False)

    # 量化模型须先量化再 load_state_dict，使权重键与结构一致
    if checkpoint.get('quantized', False):
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    if eval_mode:
        model.eval()

    info = {
        'model_type': model_type,
        'classes': checkpoint.get('classes', config.CLASSES),
        'input_size': checkpoint.get('input_size', get_input_size(model_type)),
        'accuracy': checkpoint.get('accuracy'),
        'epoch': checkpoint.get('epoch'),
        'quantized': checkpoint.get('quantized', False),
        'pruned': checkpoint.get('pruned', False),
        'distilled': checkpoint.get('distilled', False),
        'prune_amount': checkpoint.get('prune_amount'),
        'device': device,
    }
    logger.info("已加载模型 %s | 训练准确率 %s | 设备 %s", model_type, info['accuracy'], device)
    return model, info


def predict_probs(model, pil_image, transform, device):
    """对单张 PIL 图做推理，返回各类别概率列表（长度 = num_classes）"""
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return probs.cpu().tolist()
