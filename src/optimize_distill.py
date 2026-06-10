"""
optimize_distill.py - 模型优化与知识蒸馏
包含模型量化、剪枝和知识蒸馏功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time

import src.config as config
from src.dataset import EmotionDataset
from src.model import get_model
from src.utils import save_checkpoint


# ==================== 模型量化 ====================

def quantize_model(model_path, model_type, save_path=None):
    """
    动态量化模型 - 将FP32转换为INT8，减小模型大小

    优点:
        - 模型大小减少75%（约4倍压缩）
        - 推理速度提升2-4倍（CPU上）
        - 准确率损失很小（通常<1%）

    适用场景:
        - CPU部署
        - 移动端部署
        - 存储空间有限

    Args:
        model_path: 原始模型路径
        model_type: 模型类型
        save_path: 量化后模型保存路径
    """
    print("\n" + "=" * 70)
    print("🔧 模型量化（Dynamic Quantization）")
    print("=" * 70)

    # 加载原始模型
    print(f"加载模型: {model_path}")
    model = get_model(model_type, num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取原始模型大小
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

    # 动态量化
    print("\n正在量化模型...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # 量化线性层和卷积层
        dtype=torch.qint8
    )

    # 保存量化模型
    if save_path is None:
        save_path = model_path.replace('.pth', '_quantized.pth')

    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_type': model_type,
        'quantized': True
    }, save_path)

    quantized_size = os.path.getsize(save_path) / (1024 * 1024)  # MB

    print("\n" + "=" * 70)
    print("✅ 量化完成！")
    print("=" * 70)
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化后大小: {quantized_size:.2f} MB")
    print(f"压缩比: {original_size/quantized_size:.2f}x")
    print(f"保存路径: {save_path}")
    print("\n建议:")
    print("  - 在CPU上运行 evaluate.py 测试量化模型的准确率")
    print("  - 量化模型特别适合移动端和边缘设备部署")

    return quantized_model, save_path


def prune_model(model_path, model_type, prune_amount=0.3, save_path=None):
    """
    模型剪枝 - 移除不重要的权重，减小模型大小

    优点:
        - 减少参数量和计算量
        - 加快推理速度
        - 减小模型大小

    适用场景:
        - 需要轻量级模型
        - 边缘设备部署

    Args:
        model_path: 原始模型路径
        model_type: 模型类型
        prune_amount: 剪枝比例（0.3表示剪掉30%的权重）
        save_path: 剪枝后模型保存路径
    """
    print("\n" + "=" * 70)
    print(f"✂️  模型剪枝（Pruning - {prune_amount*100}%）")
    print("=" * 70)

    import torch.nn.utils.prune as prune

    # 加载原始模型
    print(f"加载模型: {model_path}")
    model = get_model(model_type, num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 统计原始参数
    original_params = sum(p.numel() for p in model.parameters())

    # 对所有卷积层和线性层进行剪枝
    print(f"\n正在剪枝 {prune_amount*100}% 的权重...")
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # 全局非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )

    # 移除剪枝的重参数化（使剪枝永久化）
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    # 统计剪枝后的参数
    pruned_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())

    # 保存剪枝模型
    if save_path is None:
        save_path = model_path.replace('.pth', f'_pruned_{int(prune_amount*100)}.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'pruned': True,
        'prune_amount': prune_amount
    }, save_path)

    print("\n" + "=" * 70)
    print("✅ 剪枝完成！")
    print("=" * 70)
    print(f"原始参数量: {original_params:,}")
    print(f"剪枝后参数量: {pruned_params:,}")
    print(f"零参数数量: {zero_params:,} ({zero_params/pruned_params*100:.2f}%)")
    print(f"保存路径: {save_path}")
    print("\n建议:")
    print("  - 剪枝后的模型需要微调（fine-tune）以恢复准确率")
    print("  - 运行 evaluate.py 测试剪枝模型的准确率")

    return model, save_path


# ==================== 知识蒸馏 ====================

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数

    结合了:
        1. 学生模型与真实标签的交叉熵损失
        2. 学生模型与教师模型输出的KL散度损失
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        """
        Args:
            temperature: 温度参数，控制软标签的平滑程度
            alpha: 平衡系数，0-1之间，控制两种损失的权重
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """
        计算蒸馏损失

        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        """
        # 硬标签损失（学生 vs 真实标签）
        hard_loss = self.ce_loss(student_logits, labels)

        # 软标签损失（学生 vs 教师）
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # 总损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss


def knowledge_distillation(
    teacher_model_path,
    teacher_model_type,
    student_model_type,
    num_epochs=15,
    batch_size=64,
    learning_rate=0.001,
    temperature=3.0,
    alpha=0.5,
    device='auto'
):
    """
    知识蒸馏训练

    将大模型（教师）的知识传递给小模型（学生）

    典型组合:
        - 教师: ResNet50/ResNet34  →  学生: ResNet18
        - 教师: ResNet18           →  学生: MobileNet
        - 教师: EfficientNet       →  学生: MobileNet

    Args:
        teacher_model_path: 教师模型路径
        teacher_model_type: 教师模型类型
        student_model_type: 学生模型类型
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        temperature: 蒸馏温度（越大越平滑）
        alpha: 硬标签损失权重（0-1）
        device: 设备
    """

    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print("\n" + "=" * 70)
    print("🎓 知识蒸馏训练")
    print("=" * 70)
    print(f"教师模型: {teacher_model_type}")
    print(f"学生模型: {student_model_type}")
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    print(f"温度参数: {temperature}")
    print(f"Alpha参数: {alpha}")
    print("=" * 70)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = EmotionDataset(str(config.TRAIN_DIR), transform=train_transform)
    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )

    # 加载教师模型
    print("\n加载教师模型...")
    teacher_model = get_model(teacher_model_type, num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 教师模型设置为评估模式，不更新参数

    print(f"✅ 教师模型加载完成")
    print(f"   验证准确率: {checkpoint.get('accuracy', 'N/A')}")

    # 创建学生模型
    print("\n创建学生模型...")
    student_model = get_model(student_model_type, num_classes=config.NUM_CLASSES, pretrained=True)
    student_model = student_model.to(device)

    # 统计参数量
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"\n模型参数对比:")
    print(f"  教师模型: {teacher_params:,} 参数")
    print(f"  学生模型: {student_params:,} 参数")
    print(f"  压缩比: {teacher_params/student_params:.2f}x")

    # 损失函数和优化器
    distillation_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # 训练历史
    history = {
        'train_loss': [],
        'train_hard_loss': [],
        'train_soft_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # 开始训练
    print("\n" + "=" * 70)
    print("开始蒸馏训练...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 训练阶段
        student_model.train()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='训练中', ncols=100)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 教师模型推理（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # 学生模型推理
            optimizer.zero_grad()
            student_logits = student_model(images)

            # 计算蒸馏损失
            loss, hard_loss, soft_loss = distillation_criterion(
                student_logits, teacher_logits, labels
            )

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()

            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        avg_loss = running_loss / len(train_loader)
        avg_hard_loss = running_hard_loss / len(train_loader)
        avg_soft_loss = running_soft_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证阶段
        student_model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='验证中', ncols=100)

            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })

        val_acc = 100 * val_correct / val_total

        # 更新学习率
        scheduler.step(val_acc)

        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_hard_loss'].append(avg_hard_loss)
        history['train_soft_loss'].append(avg_soft_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"\n结果:")
        print(f"  训练 - 总Loss: {avg_loss:.4f}, 硬Loss: {avg_hard_loss:.4f}, "
              f"软Loss: {avg_soft_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = str(config.MODELS_DIR / f'distilled_{student_model_type}_from_{teacher_model_type}.pth')
            save_checkpoint(
                student_model, optimizer, epoch+1, avg_loss, val_acc, save_path
            )
            print(f"  ✓ 新的最佳模型！验证准确率: {val_acc:.2f}%")

    # 训练完成
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ 知识蒸馏训练完成！")
    print("=" * 70)
    print(f"总训练时间: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"\n对比:")
    print(f"  教师模型原始准确率: {checkpoint.get('accuracy', 'N/A')}")
    print(f"  学生模型蒸馏后准确率: {best_val_acc:.2f}%")
    print(f"  参数量压缩比: {teacher_params/student_params:.2f}x")

    return student_model, history


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🛠️  模型优化与知识蒸馏工具")
    print("=" * 70)
    print("\n可用功能:")
    print("  [1] 模型量化（Dynamic Quantization）")
    print("  [2] 模型剪枝（Pruning）")
    print("  [3] 知识蒸馏（Knowledge Distillation）")
    print("  [4] 全部执行（量化 + 剪枝）")

    choice = input("\n请选择功能 (1-4): ").strip()

    if choice == '1':
        # 模型量化
        print("\n当前可用的模型:")
        models_dir = str(config.MODELS_DIR)
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pth')]
            for i, model in enumerate(models, 1):
                print(f"  [{i}] {model}")

            model_idx = int(input("\n选择要量化的模型编号: ")) - 1
            model_path = os.path.join(models_dir, models[model_idx])
            model_type = models[model_idx].replace('best_model_', '').replace('.pth', '')

            quantize_model(model_path, model_type)
        else:
            print("❌ models 文件夹不存在，请先训练模型")

    elif choice == '2':
        # 模型剪枝
        print("\n当前可用的模型:")
        models_dir = str(config.MODELS_DIR)
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pth')]
            for i, model in enumerate(models, 1):
                print(f"  [{i}] {model}")

            model_idx = int(input("\n选择要剪枝的模型编号: ")) - 1
            model_path = os.path.join(models_dir, models[model_idx])
            model_type = models[model_idx].replace('best_model_', '').replace('.pth', '')

            prune_amount = float(input("输入剪枝比例 (0.1-0.5，推荐0.3): "))

            prune_model(model_path, model_type, prune_amount)
        else:
            print("❌ models 文件夹不存在，请先训练模型")

    elif choice == '3':
        # 知识蒸馏
        print("\n推荐的教师-学生组合:")
        print("  [1] ResNet50 → ResNet18")
        print("  [2] ResNet34 → MobileNet")
        print("  [3] EfficientNet → MobileNet")
        print("  [4] ResNet18 → MobileNet")
        print("  [5] 自定义")

        combo_choice = input("\n选择组合 (1-5): ").strip()

        combinations = {
            '1': ('resnet50', 'resnet18'),
            '2': ('resnet34', 'mobilenet'),
            '3': ('efficientnet', 'mobilenet'),
            '4': ('resnet18', 'mobilenet')
        }

        if combo_choice in combinations:
            teacher_type, student_type = combinations[combo_choice]
        else:
            teacher_type = input("输入教师模型类型: ").strip()
            student_type = input("输入学生模型类型: ").strip()

        teacher_path = str(config.MODELS_DIR / f'best_model_{teacher_type}.pth')

        if os.path.exists(teacher_path):
            knowledge_distillation(
                teacher_model_path=teacher_path,
                teacher_model_type=teacher_type,
                student_model_type=student_type,
                num_epochs=15,
                temperature=3.0,
                alpha=0.5
            )
        else:
            print(f"❌ 教师模型不存在: {teacher_path}")
            print("请先训练教师模型")

    elif choice == '4':
        # 全部执行
        print("\n将对所有已训练的模型执行量化和剪枝...")
        models_dir = str(config.MODELS_DIR)
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pth')]

            for model_file in models:
                model_path = os.path.join(models_dir, model_file)
                model_type = model_file.replace('best_model_', '').replace('.pth', '')

                print(f"\n处理模型: {model_type}")
                quantize_model(model_path, model_type)
                prune_model(model_path, model_type, prune_amount=0.3)
        else:
            print("❌ models 文件夹不存在，请先训练模型")

    print("\n" + "=" * 70)
    print("✅ 全部完成！")
    print("=" * 70)