"""测试夹具：把一个小 torch 模型导出为临时 ONNX，供引擎测试用（避免依赖真实模型）"""

import torch
from src.model import get_model


def export_tiny_onnx(path, num_classes=5, input_size=112):
    model = get_model('cnn', num_classes=num_classes, pretrained=False).eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy, path, input_names=['input'],
                      output_names=['logits'], opset_version=13)
    return path
