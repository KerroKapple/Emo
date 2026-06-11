"""optimize_distill.py - 模型量化、剪枝与知识蒸馏"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as config
from src.logging_setup import get_logger
from src.dataset import EmotionDataset
from src.model import get_model, get_input_size, ALL_MODEL_TYPES
from src.transforms import build_transform
from src.inference import save_checkpoint, load_model_from_checkpoint, resolve_device

logger = get_logger('emotion.optimize')


def quantize_model(model_path, save_path=None):
    """动态量化（INT8）。注意：eager 模式动态量化仅作用于 nn.Linear，对 Conv2d 无效。"""
    model, info = load_model_from_checkpoint(model_path, device='cpu')
    model_type = info['model_type']
    original_mb = os.path.getsize(model_path) / (1024 * 1024)

    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    if save_path is None:
        save_path = model_path.replace('.pth', '_quantized.pth')
    save_checkpoint(quantized, model_type, save_path, val_acc=info['accuracy'], quantized=True)

    quantized_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info("量化完成 | %.2f MB -> %.2f MB (%.2fx) | %s",
                original_mb, quantized_mb, original_mb / max(quantized_mb, 1e-6), save_path)
    return quantized, save_path


def prune_model(model_path, prune_amount=0.3, save_path=None):
    """全局非结构化 L1 剪枝（永久化）"""
    model, info = load_model_from_checkpoint(model_path, device='cpu')
    model_type = info['model_type']

    to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount)
    for module, name in to_prune:
        prune.remove(module, name)

    zero = sum((p == 0).sum().item() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())

    if save_path is None:
        save_path = model_path.replace('.pth', f'_pruned_{int(prune_amount * 100)}.pth')
    save_checkpoint(model, model_type, save_path, val_acc=info['accuracy'],
                    pruned=True, extra={'prune_amount': prune_amount})
    logger.info("剪枝完成 | 零权重 %.1f%% | %s", 100 * zero / total, save_path)
    logger.info("提示: 剪枝后建议微调以恢复准确率")
    return model, save_path


class DistillationLoss(nn.Module):
    """蒸馏损失 = alpha*硬标签CE + (1-alpha)*软标签KL"""

    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.t = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        hard = self.ce(student_logits, labels)
        student_soft = nn.functional.log_softmax(student_logits / self.t, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.t, dim=1)
        soft = self.kl(student_soft, teacher_soft) * (self.t ** 2)
        return self.alpha * hard + (1 - self.alpha) * soft, hard, soft


def knowledge_distillation(teacher_model_path, student_model_type, num_epochs=15,
                           batch_size=64, learning_rate=0.001, temperature=3.0,
                           alpha=0.5, device='auto'):
    """知识蒸馏：把教师模型知识迁移到学生模型，返回 (student, history)"""
    device = resolve_device(device)
    teacher, t_info = load_model_from_checkpoint(teacher_model_path, device)
    teacher_type = t_info['model_type']

    if get_input_size(teacher_type) != get_input_size(student_model_type):
        logger.warning("师生输入尺寸不同(%s/%s)，统一按学生尺寸喂入",
                       get_input_size(teacher_type), get_input_size(student_model_type))

    transform_train = build_transform(student_model_type, train=True)
    transform_val = build_transform(student_model_type, train=False)
    train_loader = DataLoader(EmotionDataset(str(config.TRAIN_DIR), transform=transform_train),
                              batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(EmotionDataset(str(config.VAL_DIR), transform=transform_val),
                            batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=device.type == 'cuda')

    student = get_model(student_model_type, num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    logger.info("蒸馏 %s -> %s | 压缩 %.2fx", teacher_type, student_model_type, t_params / s_params)

    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)
        student.train()
        running_loss = correct = total = 0
        for images, labels in tqdm(train_loader, desc='蒸馏训练', ncols=100):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            optimizer.zero_grad()
            student_logits = student(images)
            loss, _, _ = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        student.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='验证', ncols=100):
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(student(images), 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()
        val_acc = 100 * v_correct / v_total
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        logger.info("  训练 loss=%.4f acc=%.2f%% | 验证 acc=%.2f%%", train_loss, train_acc, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = str(config.MODELS_DIR / f'distilled_{student_model_type}_from_{teacher_type}.pth')
            save_checkpoint(student, student_model_type, save_path, epoch=epoch + 1,
                            val_acc=val_acc, optimizer=optimizer, extra={'distilled': True})
            logger.info("  ✓ 新最佳: %.2f%%", val_acc)

    logger.info("蒸馏完成 | 用时 %.1f min | 最佳学生准确率 %.2f%%",
                (time.time() - start_time) / 60, best_val_acc)
    return student, history


def _parse_args():
    p = argparse.ArgumentParser(description='模型量化/剪枝/蒸馏')
    p.add_argument('--mode', required=True, choices=['quantize', 'prune', 'distill'])
    p.add_argument('--model-path', help='quantize/prune 的输入 checkpoint')
    p.add_argument('--amount', type=float, default=0.3, help='剪枝比例')
    p.add_argument('--teacher-path', help='蒸馏教师 checkpoint')
    p.add_argument('--student', choices=ALL_MODEL_TYPES, help='蒸馏学生模型类型')
    p.add_argument('--epochs', type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == 'quantize':
        quantize_model(args.model_path)
    elif args.mode == 'prune':
        prune_model(args.model_path, prune_amount=args.amount)
    elif args.mode == 'distill':
        knowledge_distillation(args.teacher_path, args.student, num_epochs=args.epochs)
