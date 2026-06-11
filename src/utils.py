"""utils.py - 工具函数：随机种子、数据划分、可视化、分类报告"""

import os
import random
import shutil
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.utils')


def set_seed(seed=42):
    """统一设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(raw_dir, train_dir, val_dir, split_ratio=0.8, seed=42, auto_clean=True):
    """将原始数据按 split_ratio 划分为训练/验证集（可选先清洗）"""
    if auto_clean:
        from src.dataset import quick_clean
        stats = quick_clean(raw_dir)
        logger.info("清洗完成: 总 %d / 有效 %d / 问题 %d",
                    stats['total_files'], stats['valid_files'],
                    stats['total_files'] - stats['valid_files'])

    random.seed(seed)
    classes = config.CLASSES

    for split_dir in (train_dir, val_dir):
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    total_train = total_val = 0
    for class_name in classes:
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_path):
            logger.warning("跳过不存在的目录: %s", class_path)
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(config.IMAGE_EXTENSIONS)]
        random.shuffle(images)
        split_point = int(len(images) * split_ratio)
        train_images, val_images = images[:split_point], images[split_point:]

        for img in tqdm(train_images, desc=f"{class_name}->train", ncols=80):
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(train_dir, class_name, img))
        for img in tqdm(val_images, desc=f"{class_name}->val", ncols=80):
            shutil.copy2(os.path.join(class_path, img),
                         os.path.join(val_dir, class_name, img))

        total_train += len(train_images)
        total_val += len(val_images)
        logger.info("%-10s 共 %d -> 训练 %d / 验证 %d",
                    class_name, len(images), len(train_images), len(val_images))

    logger.info("划分完成: 训练集 %d / 验证集 %d", total_train, total_val)


def plot_training_history(history, save_path=None):
    """绘制训练/验证 loss 与 accuracy 曲线"""
    if save_path is None:
        save_path = str(config.RESULTS_DIR / 'training_history.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("训练曲线已保存: %s", save_path)


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """绘制混淆矩阵"""
    if save_path is None:
        save_path = str(config.RESULTS_DIR / 'confusion_matrix.png')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("混淆矩阵已保存: %s", save_path)


def classification_report_text(y_true, y_pred, classes):
    """返回并记录分类报告文本"""
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    logger.info("分类报告:\n%s", report)
    return report


def _parse_args():
    p = argparse.ArgumentParser(description='清洗并划分原始数据为训练/验证集')
    p.add_argument('--ratio', type=float, default=0.8, help='训练集比例')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-clean', action='store_true', help='划分前不清洗')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_dataset(str(config.RAW_DIR), str(config.TRAIN_DIR), str(config.VAL_DIR),
                  split_ratio=args.ratio, seed=args.seed, auto_clean=not args.no_clean)
