"""
model.py - 表情识别CNN模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmotionCNN(nn.Module):
    """
    自定义卷积神经网络用于表情识别

    架构说明:
        - 3个卷积块，每块包含2个卷积层 + BatchNorm + ReLU + MaxPool + Dropout
        - 逐层增加特征图数量: 64 -> 128 -> 256
        - 使用MaxPool逐步降低空间维度: 48x48 -> 24x24 -> 12x12 -> 6x6
        - 最后用全连接层进行分类

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5):
        super(EmotionCNN, self).__init__()

        # 第一个卷积块: 提取低级特征（边缘、纹理等）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入3通道(RGB) -> 输出64通道
            nn.BatchNorm2d(64),                          # 批归一化，加速训练
            nn.ReLU(inplace=True),                       # 激活函数，增加非线性
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 再次卷积，加深特征提取
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 池化，降维 48x48 -> 24x24
            nn.Dropout(0.25)                             # 随机丢弃25%神经元，防止过拟合
        )

        # 第二个卷积块: 提取中级特征（形状、局部模式等）
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64通道 -> 128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 24x24 -> 12x12
            nn.Dropout(0.25)
        )

        # 第三个卷积块: 提取高级特征（复杂模式、语义信息等）
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128通道 -> 256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 12x12 -> 6x6
            nn.Dropout(0.25)
        )

        # 全连接层: 将特征映射到类别
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),  # 展平后: 9216维 -> 512维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # 更高的dropout率，进一步防止过拟合
            nn.Linear(512, num_classes)   # 512维 -> 5个类别
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        x = self.conv1(x)                # -> (batch_size, 64, 24, 24)
        x = self.conv2(x)                # -> (batch_size, 128, 12, 12)
        x = self.conv3(x)                # -> (batch_size, 256, 6, 6)
        x = x.view(x.size(0), -1)        # 展平 -> (batch_size, 9216)
        x = self.fc(x)                   # -> (batch_size, 5)
        return x


# 迁移学习模型注册表：model_type -> (构造函数, 权重枚举, head 类型)
# head 类型: 'fc'=替换 .fc; 'classifier_last'=替换 .classifier[1]; 'vgg'=替换 .classifier[6]
_TRANSFER_REGISTRY = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, 'fc'),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT, 'fc'),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, 'fc'),
    'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT, 'vgg'),
    'mobilenet': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, 'classifier_last'),
    'efficientnet': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 'classifier_last'),
}


class TransferModel(nn.Module):
    """统一的迁移学习包装：加载预训练 backbone 并替换分类头为 num_classes"""

    def __init__(self, model_type, num_classes=5, pretrained=True):
        super().__init__()
        builder, weights_enum, head_kind = _TRANSFER_REGISTRY[model_type]
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

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type='cnn', num_classes=5, pretrained=True):
    """模型工厂：'cnn' 用自定义 CNN，其余走迁移学习注册表"""
    if model_type == 'cnn':
        return EmotionCNN(num_classes=num_classes)
    if model_type in _TRANSFER_REGISTRY:
        return TransferModel(model_type, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"不支持的模型类型: {model_type}")


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    return total_params, trainable_params


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("测试所有可用模型")
    print("=" * 70)

    models_to_test = ['cnn', 'resnet18', 'mobilenet', 'efficientnet']

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"测试 {model_name.upper()} 模型")
        print('='*70)

        try:
            model = get_model(model_name, num_classes=5, pretrained=False)
            count_parameters(model)

            # 测试前向传播
            dummy_input = torch.randn(2, 3, 48, 48)
            output = model(dummy_input)
            print(f"\n输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"✅ {model_name.upper()} 测试成功！")

        except Exception as e:
            print(f"❌ {model_name.upper()} 测试失败: {e}")

    print("\n" + "=" * 70)
    print("模型测试完成！")
    print("=" * 70)
