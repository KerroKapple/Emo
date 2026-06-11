"""evaluate.py - 模型评估（支持普通/量化/剪枝/蒸馏模型）"""

import os
import time
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as config
from src.logging_setup import get_logger
from src.dataset import EmotionDataset
from src.transforms import build_transform
from src.inference import load_model_from_checkpoint, resolve_device, predict_probs
from src.utils import plot_confusion_matrix, classification_report_text

logger = get_logger('emotion.evaluate')


def evaluate_model(model_path, batch_size=64, device='auto', save_results=True):
    """评估单个 checkpoint，返回结果字典"""
    device = resolve_device(device)
    model, info = load_model_from_checkpoint(model_path, device)
    model_type = info['model_type']

    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=build_transform(model_type, train=False))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=device.type == 'cuda')

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    all_pred, all_true, inference_times = [], [], []
    correct = total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='评估中', ncols=100):
            images, labels = images.to(device), labels.to(device)
            t0 = time.time()
            outputs = model(images)
            inference_times.append(time.time() - t0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_batch_ms = float(np.mean(inference_times)) * 1000
    throughput = batch_size / float(np.mean(inference_times))

    logger.info("评估 %s | 准确率 %.2f%% | 大小 %.2f MB | %.2f ms/batch | %.1f img/s",
                os.path.basename(model_path), accuracy, model_size_mb, avg_batch_ms, throughput)
    classification_report_text(all_true, all_pred, config.CLASSES)

    if save_results:
        name = os.path.basename(model_path).replace('.pth', '')
        plot_confusion_matrix(all_true, all_pred, config.CLASSES,
                              str(config.RESULTS_DIR / f'confusion_matrix_{name}.png'))

    for i, cls in enumerate(config.CLASSES):
        idx = [j for j, t in enumerate(all_true) if t == i]
        if idx:
            acc = 100 * sum(all_pred[j] == i for j in idx) / len(idx)
            logger.info("  %-10s %.2f%% (%d)", cls, acc, len(idx))

    return {
        'model_name': os.path.basename(model_path),
        'model_type': model_type,
        'accuracy': accuracy,
        'model_size_mb': model_size_mb,
        'inference_time_ms': avg_batch_ms,
        'throughput': throughput,
        'quantized': info['quantized'],
        'pruned': info['pruned'],
        'distilled': info['distilled'],
    }


def evaluate_all_models():
    """评估 models/ 下所有 checkpoint 并生成对比报告"""
    models_dir = str(config.MODELS_DIR)
    if not os.path.exists(models_dir):
        logger.error("models 目录不存在，请先训练")
        return []

    results = []
    for f in sorted(os.listdir(models_dir)):
        if not f.endswith('.pth'):
            continue
        try:
            results.append(evaluate_model(os.path.join(models_dir, f), save_results=True))
        except Exception as e:
            logger.error("评估失败 %s: %s", f, e)
    _comprehensive_report(results)
    return results


def _comprehensive_report(results):
    """打印并保存综合对比报告"""
    ok = [r for r in results if 'accuracy' in r]
    if not ok:
        logger.warning("没有可用的评估结果")
        return

    df = pd.DataFrame(ok)
    cols = ['model_name', 'accuracy', 'model_size_mb', 'inference_time_ms', 'throughput']
    logger.info("模型对比:\n%s", df[cols].to_string(index=False))

    best_acc = df.loc[df['accuracy'].idxmax()]
    smallest = df.loc[df['model_size_mb'].idxmin()]
    fastest = df.loc[df['inference_time_ms'].idxmin()]
    df['efficiency'] = df['accuracy'] / df['model_size_mb']
    efficient = df.loc[df['efficiency'].idxmax()]
    logger.info("最高准确率: %s (%.2f%%)", best_acc['model_name'], best_acc['accuracy'])
    logger.info("最小模型: %s (%.2f MB)", smallest['model_name'], smallest['model_size_mb'])
    logger.info("最快推理: %s (%.2f ms)", fastest['model_name'], fastest['inference_time_ms'])
    logger.info("最高效率: %s (%.2f)", efficient['model_name'], efficient['efficiency'])

    csv_path = str(config.RESULTS_DIR / 'comprehensive_evaluation.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info("详细报告已保存: %s", csv_path)


def predict_single_image(model_path, image_path, device='auto'):
    """对单张图片预测并打印各类别概率"""
    from PIL import Image

    device = resolve_device(device)
    model, info = load_model_from_checkpoint(model_path, device)
    transform = build_transform(info['model_type'], train=False)
    image = Image.open(image_path).convert('RGB')

    t0 = time.time()
    probs = predict_probs(model, image, transform, device)
    elapsed_ms = (time.time() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    logger.info("图片 %s | 预测 %s | 置信度 %.2f%% | %.2f ms",
                image_path, config.CLASSES[pred_idx], probs[pred_idx] * 100, elapsed_ms)
    for cls, p in zip(config.CLASSES, probs):
        logger.info("  %-10s %5.2f%%", cls, p * 100)
    return config.CLASSES[pred_idx], probs[pred_idx]


def _parse_args():
    p = argparse.ArgumentParser(description='评估表情识别模型')
    p.add_argument('--model-path', help='单个 checkpoint 路径')
    p.add_argument('--image', help='与 --model-path 搭配，预测单张图片')
    p.add_argument('--all', action='store_true', help='评估 models/ 下所有模型')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = 'cpu' if args.cpu else 'auto'

    if args.model_path and args.image:
        predict_single_image(args.model_path, args.image, device=device)
    elif args.model_path:
        evaluate_model(args.model_path, device=device)
    else:
        evaluate_all_models()
