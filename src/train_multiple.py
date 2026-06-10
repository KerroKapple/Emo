"""
train_multiple.py - 批量训练多个模型并对比结果（带模型选择和进度可视化）
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import src.config as config
from src.train import train_model
import time
import os
from datetime import datetime


def select_models():
    """
    交互式选择要训练的模型

    Returns:
        selected_configs: 选中的模型配置列表
    """
    # 所有可用的模型配置
    all_models = [
        {
            'id': 1,
            'name': 'ResNet18',
            'model_type': 'resnet18',
            'num_epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'description': '最推荐 - 平衡性最好',
            'time_estimate': '20-30分钟(GPU) / 2-3小时(CPU)'
        },
        {
            'id': 2,
            'name': 'MobileNet',
            'model_type': 'mobilenet',
            'num_epochs': 20,
            'batch_size': 128,
            'learning_rate': 0.001,
            'description': '最快 - 轻量级模型',
            'time_estimate': '15-20分钟(GPU) / 1-2小时(CPU)'
        },
        {
            'id': 3,
            'name': 'EfficientNet',
            'model_type': 'efficientnet',
            'num_epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'description': '最先进 - 最佳平衡',
            'time_estimate': '25-35分钟(GPU) / 3-4小时(CPU)'
        },
        {
            'id': 4,
            'name': 'ResNet34',
            'model_type': 'resnet34',
            'num_epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'description': '更深 - 更高准确率',
            'time_estimate': '30-40分钟(GPU) / 3-5小时(CPU)'
        },
        {
            'id': 5,
            'name': 'ResNet50',
            'model_type': 'resnet50',
            'num_epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'description': '最强 - 最高准确率',
            'time_estimate': '35-50分钟(GPU) / 4-6小时(CPU)'
        },
        {
            'id': 6,
            'name': 'VGG16',
            'model_type': 'vgg16',
            'num_epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'description': '经典 - CNN架构',
            'time_estimate': '40-60分钟(GPU) / 5-8小时(CPU)'
        },
        {
            'id': 7,
            'name': 'Custom CNN',
            'model_type': 'cnn',
            'num_epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.001,
            'description': '自定义 - 从零训练',
            'time_estimate': '40-60分钟(GPU) / 4-6小时(CPU)'
        }
    ]

    print("\n" + "=" * 80)
    print("可用模型列表")
    print("=" * 80)
    for model in all_models:
        print(f"  [{model['id']}] {model['name']:15s} - {model['description']}")
        print(f"      预计训练时间: {model['time_estimate']}")
    print("=" * 80)

    print("\n选择模式:")
    print("  [0] 训练所有模型")
    print("  [1-7] 训练单个模型")
    print("  输入多个数字（用逗号或空格分隔）训练多个模型，例如: 1,2,3 或 1 2 3")

    while True:
        choice = input("\n请输入你的选择: ").strip()

        if choice == '0':
            print("\n✅ 已选择: 训练所有7个模型")
            return all_models

        # 解析输入
        try:
            # 支持逗号或空格分隔
            if ',' in choice:
                selected_ids = [int(x.strip()) for x in choice.split(',')]
            else:
                selected_ids = [int(x.strip()) for x in choice.split()]

            # 验证ID有效性
            if all(1 <= id <= 7 for id in selected_ids):
                selected_models = [m for m in all_models if m['id'] in selected_ids]

                print(f"\n✅ 已选择 {len(selected_models)} 个模型:")
                for model in selected_models:
                    print(f"  - {model['name']}")

                confirm = input("\n确认开始训练? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_models
                else:
                    print("已取消，请重新选择")
            else:
                print("❌ 无效输入！请输入1-7之间的数字")
        except ValueError:
            print("❌ 输入格式错误！请输入数字")


def print_progress_bar(current, total, bar_length=50):
    """
    打印ASCII进度条

    Args:
        current: 当前进度
        total: 总数
        bar_length: 进度条长度
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f"\r总体进度: [{bar}] {current}/{total} ({percent*100:.1f}%)", end='', flush=True)


def visualize_realtime_progress(results, current_model_name):
    """
    实时可视化训练进度

    Args:
        results: 当前所有结果
        current_model_name: 当前正在训练的模型名称
    """
    if not results:
        return

    # 只显示成功的结果
    successful_results = [r for r in results if r.get('status') == 'Success']

    if not successful_results:
        return

    df = pd.DataFrame(successful_results)

    # 创建实时进度图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 准确率进度
    ax1 = axes[0]
    colors = ['green' if acc >= 80 else 'orange' if acc >= 70 else 'red'
              for acc in df['best_val_acc']]
    bars1 = ax1.barh(df['model'], df['best_val_acc'], color=colors, alpha=0.7)
    ax1.set_xlabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Current Training Progress - Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for i, (bar, acc) in enumerate(zip(bars1, df['best_val_acc'])):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)

    # 2. 训练时间
    ax2 = axes[1]
    bars2 = ax2.barh(df['model'], df['training_time_min'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Training Time (minutes)', fontsize=11, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, (bar, time_min) in enumerate(zip(bars2, df['training_time_min'])):
        ax2.text(time_min + 1, bar.get_y() + bar.get_height()/2,
                f'{time_min:.1f}m', va='center', fontsize=10)

    plt.tight_layout()

    # 保存实时进度图
    save_path = str(config.RESULTS_DIR / 'training_progress_realtime.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_all_models(selected_models):
    """
    训练选中的模型并保存对比结果

    Args:
        selected_models: 选中的模型配置列表
    """

    print("\n" + "=" * 80)
    print("批量训练模型对比实验")
    print("=" * 80)
    print(f"将训练 {len(selected_models)} 个模型:")
    for i, config in enumerate(selected_models, 1):
        print(f"  {i}. {config['name']:15s} - {config['description']}")
    print("=" * 80)

    # 记录开始时间
    total_start_time = time.time()

    # 存储所有模型的结果
    results = []

    # 开始训练每个模型
    for i, config in enumerate(selected_models, 1):
        print(f"\n\n{'='*80}")
        print(f"[{i}/{len(selected_models)}] 开始训练: {config['name']}")
        print(f"预计时间: {config['time_estimate']}")
        print(f"{'='*80}\n")

        # 更新总体进度条
        print_progress_bar(i-1, len(selected_models))
        print()  # 换行

        start_time = time.time()

        try:
            # 训练模型
            model, history = train_model(
                model_type=config['model_type'],
                num_epochs=config['num_epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                device='auto'
            )

            training_time = time.time() - start_time

            # 记录结果
            result = {
                'model': config['name'],
                'model_type': config['model_type'],
                'epochs': config['num_epochs'],
                'batch_size': config['batch_size'],
                'best_train_acc': max(history['train_acc']),
                'best_val_acc': max(history['val_acc']),
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'training_time_min': training_time / 60,
                'training_time_sec': training_time,
                'status': 'Success',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            results.append(result)

            print(f"\n{'='*80}")
            print(f"✅ {config['name']} 训练完成!")
            print(f"{'='*80}")
            print(f"   最佳验证准确率: {result['best_val_acc']:.2f}%")
            print(f"   训练时间: {result['training_time_min']:.2f} 分钟")
            print(f"   完成时间: {result['timestamp']}")

            # 保存中间结果
            save_results_table(results)

            # 实时可视化进度
            visualize_realtime_progress(results, config['name'])

        except Exception as e:
            print(f"\n❌ {config['name']} 训练失败: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'model': config['name'],
                'model_type': config['model_type'],
                'status': f'Failed: {str(e)}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(result)

    # 更新最终进度条
    print_progress_bar(len(selected_models), len(selected_models))
    print("\n")

    # 计算总时间
    total_time = time.time() - total_start_time

    # 训练完成，生成对比报告
    print("\n" + "=" * 80)
    print("所有模型训练完成！")
    print("=" * 80)
    print(f"总训练时间: {total_time/3600:.2f} 小时 ({total_time/60:.2f} 分钟)")

    generate_comparison_report(results)

    return results


def save_results_table(results):
    """保存结果表格"""
    df = pd.DataFrame(results)

    # 保存为CSV
    csv_path = str(config.RESULTS_DIR / 'models_comparison.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n💾 结果已保存至: {csv_path}")


def generate_comparison_report(results):
    """生成模型对比报告"""

    # 只保留成功的模型
    successful_results = [r for r in results if r.get('status') == 'Success']

    if not successful_results:
        print("\n⚠️  没有成功训练的模型，无法生成对比报告")
        return

    df = pd.DataFrame(successful_results)

    print("\n" + "=" * 80)
    print("📊 模型性能对比")
    print("=" * 80)
    print(df[['model', 'best_val_acc', 'final_val_acc', 'training_time_min']].to_string(index=False))

    # 找出最佳模型
    best_acc_model = df.loc[df['best_val_acc'].idxmax()]
    fastest_model = df.loc[df['training_time_min'].idxmin()]

    # 计算效率最高的模型（准确率/时间）
    df['efficiency'] = df['best_val_acc'] / df['training_time_min']
    most_efficient = df.loc[df['efficiency'].idxmax()]

    print("\n" + "=" * 80)
    print("🏆 关键指标")
    print("=" * 80)
    print(f"🥇 最高准确率: {best_acc_model['model']} - {best_acc_model['best_val_acc']:.2f}%")
    print(f"⚡ 最快训练: {fastest_model['model']} - {fastest_model['training_time_min']:.2f} 分钟")
    print(f"💡 最高效率: {most_efficient['model']} - {most_efficient['efficiency']:.2f} (准确率%/分钟)")

    # 绘制对比图表
    plot_comparison_charts(df)


def plot_comparison_charts(df):
    """绘制对比图表"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 设置颜色方案
    colors = plt.cm.Set3(range(len(df)))

    # 1. 验证准确率对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['model'], df['best_val_acc'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # 在柱子上添加数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. 训练时间对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model'], df['training_time_min'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 效率对比 (准确率 / 训练时间)
    ax3 = axes[1, 0]
    efficiency = df['best_val_acc'] / df['training_time_min']
    bars3 = ax3.bar(df['model'], efficiency, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Efficiency (Acc% / min)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. 训练损失 vs 验证损失
    ax4 = axes[1, 1]
    x_pos = range(len(df))
    width = 0.35
    bars4a = ax4.bar([p - width/2 for p in x_pos], df['final_train_loss'],
                     width, label='Train Loss', color='lightblue', alpha=0.8, edgecolor='black')
    bars4b = ax4.bar([p + width/2 for p in x_pos], df['final_val_loss'],
                     width, label='Val Loss', color='lightcoral', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax4.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df['model'], rotation=45, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = str(config.RESULTS_DIR / 'models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图表已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 欢迎使用批量训练脚本（增强版）")
    print("=" * 80)
    print("\n功能特点:")
    print("  ✅ 自由选择要训练的模型")
    print("  ✅ 实时进度可视化")
    print("  ✅ 自动保存中间结果")
    print("  ✅ 详细的性能对比报告")

    # 检测设备
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"\n当前设备: {device}")

    if device == 'CPU':
        print("\n⚠️  警告: 未检测到GPU，训练速度会较慢")
        print("   建议: 先选择1-2个轻量级模型（如MobileNet）进行测试")

    # 选择模型
    selected_models = select_models()

    # 确认开始
    print("\n" + "=" * 80)
    print("准备开始训练...")
    print("=" * 80)
    input("\n按 Enter 键开始训练...")

    # 开始训练
    results = train_all_models(selected_models)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)
    print("\n查看结果:")
    print("  📄 详细数据: results/models_comparison.csv")
    print("  📊 对比图表: results/models_comparison.png")
    print("  📈 实时进度: results/training_progress_realtime.png")
    print("  📉 各模型训练曲线: results/training_history_*.png")
    print("  💾 各模型权重: models/best_model_*.pth")
    print("\n下一步: 运行 python evaluate.py 评估模型性能")