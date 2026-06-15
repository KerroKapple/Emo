"""train.py - 单模型训练"""

import time
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as config
from src.logging_setup import get_logger
from src.dataset import EmotionDataset
from src.model import get_model, count_parameters, ALL_MODEL_TYPES
from src.transforms import build_transform
from src.inference import save_checkpoint, resolve_device
from src.utils import set_seed, plot_training_history

logger = get_logger('emotion.train')


def compute_class_weights(dataset, device):
    """按类别频次的逆频率计算交叉熵权重"""
    dist = dataset.get_class_distribution()
    counts = torch.tensor([dist[c] for c in config.CLASSES], dtype=torch.float)
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def _run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    """跑一个 epoch；train=False 时只前向。返回 (avg_loss, accuracy)"""
    model.train(train)
    use_amp = scaler is not None
    running_loss = correct = total = 0
    desc = '训练中' if train else '验证中'

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, desc=desc, ncols=100):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


def train_model(model_type='resnet18', num_epochs=20, batch_size=64, learning_rate=0.001,
                weight_decay=1e-4, label_smoothing=0.05, seed=42, num_workers=4,
                early_stop_patience=7, use_amp=True, use_class_weights=True, device='auto'):
    """完整训练流程，返回 (model, history)"""
    set_seed(seed)
    device = resolve_device(device)
    use_amp = use_amp and device.type == 'cuda'

    logger.info("开始训练 | 模型=%s 设备=%s epochs=%d batch=%d lr=%s amp=%s",
                model_type, device, num_epochs, batch_size, learning_rate, use_amp)

    train_dataset = EmotionDataset(str(config.TRAIN_DIR), transform=build_transform(model_type, train=True))
    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=build_transform(model_type, train=False))

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    model = get_model(model_type, num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    count_parameters(model)

    weights = compute_class_weights(train_dataset, device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, None, train=False)
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        logger.info("  训练 loss=%.4f acc=%.2f%% | 验证 loss=%.4f acc=%.2f%%",
                    train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, model_type, str(config.MODELS_DIR / f'best_model_{model_type}.pth'),
                            epoch=epoch + 1, val_loss=val_loss, val_acc=val_acc, optimizer=optimizer)
            logger.info("  ✓ 新最佳: %.2f%%", val_acc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("  早停: 连续 %d epoch 无提升", early_stop_patience)
                break

    total_time = time.time() - start_time
    logger.info("训练完成 | 用时 %.1f min | 最佳验证准确率 %.2f%%", total_time / 60, best_val_acc)

    save_checkpoint(model, model_type, str(config.MODELS_DIR / f'final_model_{model_type}.pth'),
                    epoch=len(history['val_acc']), val_loss=history['val_loss'][-1],
                    val_acc=history['val_acc'][-1], optimizer=optimizer)
    plot_training_history(history, str(config.RESULTS_DIR / f'training_history_{model_type}.png'))
    return model, history


def _parse_args():
    p = argparse.ArgumentParser(description='训练单个表情识别模型')
    p.add_argument('--model', default='efficientnet', choices=ALL_MODEL_TYPES, dest='model_type')
    p.add_argument('--epochs', type=int, default=None, dest='num_epochs')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None, dest='learning_rate')
    p.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = config.TrainConfig(model_type=args.model_type)
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.cpu:
        cfg.device = 'cpu'

    logger.info("训练配置: %s", asdict(cfg))
    train_model(**asdict(cfg))
