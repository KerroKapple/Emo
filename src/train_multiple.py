"""train_multiple.py - 批量训练多个模型并生成对比报告"""

import os
import time
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import src.config as config
from src.logging_setup import get_logger
from src.model import ALL_MODEL_TYPES
from src.train import train_model

logger = get_logger('emotion.train_multiple')

# 各模型的默认批量训练规格
MODEL_SPECS = {
    'resnet18': {'name': 'ResNet18', 'num_epochs': 20, 'batch_size': 64},
    'mobilenet': {'name': 'MobileNetV2', 'num_epochs': 20, 'batch_size': 128},
    'efficientnet': {'name': 'EfficientNet-B0', 'num_epochs': 20, 'batch_size': 64},
    'resnet34': {'name': 'ResNet34', 'num_epochs': 20, 'batch_size': 64},
    'resnet50': {'name': 'ResNet50', 'num_epochs': 20, 'batch_size': 32},
    'vgg16': {'name': 'VGG16', 'num_epochs': 20, 'batch_size': 32},
    'cnn': {'name': 'Custom CNN', 'num_epochs': 30, 'batch_size': 64},
}


def train_all_models(model_types):
    """依次训练指定模型，返回结果列表"""
    results = []
    total_start = time.time()

    for i, model_type in enumerate(model_types, 1):
        spec = MODEL_SPECS[model_type]
        logger.info("[%d/%d] 训练 %s", i, len(model_types), spec['name'])
        start = time.time()
        try:
            _, history = train_model(model_type=model_type, num_epochs=spec['num_epochs'],
                                     batch_size=spec['batch_size'], device='auto')
            results.append({
                'model': spec['name'],
                'model_type': model_type,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'training_time_min': (time.time() - start) / 60,
                'status': 'Success',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            _save_results(results)
        except Exception as e:
            logger.error("%s 训练失败: %s", spec['name'], e)
            results.append({'model': spec['name'], 'model_type': model_type, 'status': f'Failed: {e}'})

    logger.info("全部完成 | 总用时 %.2f 小时", (time.time() - total_start) / 3600)
    _comparison_report(results)
    return results


def _save_results(results):
    df = pd.DataFrame(results)
    csv_path = str(config.RESULTS_DIR / 'models_comparison.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def _comparison_report(results):
    ok = [r for r in results if r.get('status') == 'Success']
    if not ok:
        logger.warning("没有成功训练的模型")
        return

    df = pd.DataFrame(ok)
    logger.info("对比:\n%s", df[['model', 'best_val_acc', 'final_val_acc', 'training_time_min']].to_string(index=False))
    best = df.loc[df['best_val_acc'].idxmax()]
    fastest = df.loc[df['training_time_min'].idxmin()]
    logger.info("最高准确率: %s (%.2f%%)", best['model'], best['best_val_acc'])
    logger.info("最快训练: %s (%.2f min)", fastest['model'], fastest['training_time_min'])
    _plot_comparison(df)


def _plot_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(df['model'], df['best_val_acc'], color='steelblue', alpha=0.8)
    axes[0].set(ylabel='Best Val Acc (%)', title='Accuracy', ylim=(0, 100))
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].bar(df['model'], df['training_time_min'], color='indianred', alpha=0.8)
    axes[1].set(ylabel='Time (min)', title='Training Time')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_path = str(config.RESULTS_DIR / 'models_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("对比图已保存: %s", save_path)


def _parse_args():
    p = argparse.ArgumentParser(description='批量训练多个模型并对比')
    p.add_argument('--models', nargs='+', choices=ALL_MODEL_TYPES,
                   help='要训练的模型类型，省略则训练全部')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    selected = args.models if args.models else list(MODEL_SPECS)
    logger.info("将训练: %s", selected)
    train_all_models(selected)
