"""
evaluate.py - 模型评估脚本（支持优化和蒸馏后的模型）
评估训练好的模型在验证集上的性能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import time
import pandas as pd

import src.config as config
from src.dataset import EmotionDataset
from src.model import get_model
from src.utils import (
    plot_confusion_matrix,
    print_classification_report,
    load_checkpoint
)


def load_model_for_evaluation(model_path, model_type, device):
    """
    加载模型用于评估（支持普通、量化、剪枝、蒸馏模型）

    Args:
        model_path: 模型权重文件路径
        model_type: 模型类型
        device: 设备

    Returns:
        model: 加载好的模型
        model_info: 模型信息字典
    """
    print(f"\n加载模型: {model_path}")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 判断模型类型
    is_quantized = checkpoint.get('quantized', False)
    is_pruned = checkpoint.get('pruned', False)
    is_distilled = 'distilled' in model_path

    # 创建模型
    model = get_model(model_type, num_classes=config.NUM_CLASSES, pretrained=False)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 如果是量化模型，执行量化
    if is_quantized:
        print("✓ 检测到量化模型")
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

    model = model.to(device)
    model.eval()

    # 收集模型信息
    model_info = {
        'model_type': model_type,
        'quantized': is_quantized,
        'pruned': is_pruned,
        'distilled': is_distilled,
        'saved_accuracy': checkpoint.get('accuracy', 'N/A'),
        'saved_epoch': checkpoint.get('epoch', 'N/A'),
        'prune_amount': checkpoint.get('prune_amount', 'N/A')
    }

    print(f"✅ 模型加载成功")
    print(f"   模型类型: {model_type}")
    if is_quantized:
        print(f"   ✓ 量化模型")
    if is_pruned:
        print(f"   ✓ 剪枝模型 (剪枝比例: {checkpoint.get('prune_amount', 'N/A')})")
    if is_distilled:
        print(f"   ✓ 蒸馏模型")
    print(f"   训练时准确率: {checkpoint.get('accuracy', 'N/A')}")

    return model, model_info


def evaluate_model(model_path, model_type, batch_size=64, device='auto', save_results=True):
    """
    评估模型性能

    Args:
        model_path: 模型权重文件路径
        model_type: 模型类型
        batch_size: 批次大小
        device: 设备
        save_results: 是否保存结果

    Returns:
        results: 评估结果字典
    """

    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print("=" * 70)
    print("📊 模型评估")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"设备: {device}")
    print("=" * 70)

    # 数据预处理（与验证集相同）
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN,
                           std=config.IMAGENET_STD)
    ])

    # 加载验证集
    print("\n加载验证集...")
    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 加载模型
    model, model_info = load_model_for_evaluation(model_path, model_type, device)

    # 获取模型大小
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # 评估
    print("\n" + "=" * 70)
    print("开始评估...")
    print("=" * 70)

    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    # 测量推理时间
    inference_times = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='评估中', ncols=100)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 测量推理时间
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            _, predicted = torch.max(outputs.data, 1)

            # 统计
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            accuracy = 100 * correct / total
            pbar.set_postfix({'acc': f'{accuracy:.2f}%'})

    # 计算最终准确率
    final_accuracy = 100 * correct / total

    # 计算推理速度
    avg_inference_time = np.mean(inference_times)
    avg_batch_time = avg_inference_time * 1000  # 转换为毫秒
    throughput = batch_size / avg_inference_time  # 图片/秒

    print("\n" + "=" * 70)
    print("📈 评估结果")
    print("=" * 70)
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"准确率: {final_accuracy:.2f}%")
    print(f"\n性能指标:")
    print(f"  模型大小: {model_size_mb:.2f} MB")
    print(f"  推理速度: {avg_batch_time:.2f} ms/batch")
    print(f"  吞吐量: {throughput:.2f} 图片/秒")

    # 类别名称
    classes = config.CLASSES

    # 打印分类报告
    print_classification_report(all_labels, all_predictions, classes)

    # 保存混淆矩阵
    if save_results:
        # 生成保存路径（包含模型特征）
        model_name = os.path.basename(model_path).replace('.pth', '')
        cm_path = str(config.RESULTS_DIR / f'confusion_matrix_{model_name}.png')
        plot_confusion_matrix(all_labels, all_predictions, classes, cm_path)

    # 计算每个类别的准确率
    print("\n" + "=" * 70)
    print("📊 各类别准确率")
    print("=" * 70)

    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    for label, pred in zip(all_labels, all_predictions):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_name:10s}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    print("\n" + "=" * 70)
    print("✅ 评估完成！")
    print("=" * 70)

    # 返回结果
    results = {
        'model_path': model_path,
        'model_type': model_type,
        'accuracy': final_accuracy,
        'model_size_mb': model_size_mb,
        'inference_time_ms': avg_batch_time,
        'throughput': throughput,
        'quantized': model_info['quantized'],
        'pruned': model_info['pruned'],
        'distilled': model_info['distilled'],
        'prune_amount': model_info.get('prune_amount', 'N/A')
    }

    return results


def evaluate_all_models():
    """评估所有训练好的模型（包括优化和蒸馏后的）"""

    models_dir = str(config.MODELS_DIR)

    if not os.path.exists(models_dir):
        print("❌ models 文件夹不存在，请先训练模型")
        return

    # 获取所有模型文件
    all_model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

    # 分类模型
    base_models = [f for f in all_model_files if f.startswith('best_model_')]
    quantized_models = [f for f in all_model_files if 'quantized' in f]
    pruned_models = [f for f in all_model_files if 'pruned' in f]
    distilled_models = [f for f in all_model_files if 'distilled' in f]

    print("\n" + "=" * 70)
    print("📊 批量评估所有模型")
    print("=" * 70)
    print(f"\n发现模型:")
    print(f"  基础模型: {len(base_models)} 个")
    print(f"  量化模型: {len(quantized_models)} 个")
    print(f"  剪枝模型: {len(pruned_models)} 个")
    print(f"  蒸馏模型: {len(distilled_models)} 个")
    print(f"  总计: {len(all_model_files)} 个")

    all_results = []

    # 评估所有模型
    for model_file in all_model_files:
        model_path = os.path.join(models_dir, model_file)

        # 推断模型类型
        if 'distilled' in model_file:
            # 蒸馏模型格式: distilled_studenttype_from_teachertype.pth
            parts = model_file.replace('distilled_', '').replace('.pth', '').split('_from_')
            model_type = parts[0]
        elif 'quantized' in model_file:
            model_type = model_file.replace('best_model_', '').replace('_quantized.pth', '')
        elif 'pruned' in model_file:
            model_type = model_file.replace('best_model_', '').split('_pruned_')[0]
        else:
            model_type = model_file.replace('best_model_', '').replace('final_model_', '').replace('.pth', '')

        try:
            print(f"\n{'='*70}")
            print(f"评估: {model_file}")
            print(f"{'='*70}")

            result = evaluate_model(model_path, model_type, save_results=True)
            result['model_name'] = model_file
            all_results.append(result)

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            all_results.append({
                'model_name': model_file,
                'model_type': model_type,
                'status': f'Failed: {e}'
            })

    # 生成对比报告
    generate_comprehensive_report(all_results)

    return all_results


def generate_comprehensive_report(results):
    """生成综合对比报告"""

    # 只保留成功的结果
    successful_results = [r for r in results if 'accuracy' in r]

    if not successful_results:
        print("\n⚠️  没有成功评估的模型")
        return

    df = pd.DataFrame(successful_results)

    # 打印详细对比表
    print("\n" + "=" * 80)
    print("📊 所有模型性能对比")
    print("=" * 80)

    display_cols = ['model_name', 'accuracy', 'model_size_mb', 'inference_time_ms', 'throughput']
    print(df[display_cols].to_string(index=False))

    # 按类别分组统计
    print("\n" + "=" * 80)
    print("📈 分类统计")
    print("=" * 80)

    # 基础模型
    base_df = df[~df['quantized'] & ~df['pruned'] & ~df['distilled']]
    if not base_df.empty:
        print(f"\n🔹 基础模型 ({len(base_df)}个):")
        print(f"   平均准确率: {base_df['accuracy'].mean():.2f}%")
        print(f"   平均大小: {base_df['model_size_mb'].mean():.2f} MB")
        print(f"   平均速度: {base_df['inference_time_ms'].mean():.2f} ms/batch")

    # 量化模型
    quant_df = df[df['quantized']]
    if not quant_df.empty:
        print(f"\n🔹 量化模型 ({len(quant_df)}个):")
        print(f"   平均准确率: {quant_df['accuracy'].mean():.2f}%")
        print(f"   平均大小: {quant_df['model_size_mb'].mean():.2f} MB")
        print(f"   平均速度: {quant_df['inference_time_ms'].mean():.2f} ms/batch")
        if not base_df.empty:
            print(f"   📉 大小压缩: {base_df['model_size_mb'].mean() / quant_df['model_size_mb'].mean():.2f}x")
            print(f"   ⚡ 速度提升: {base_df['inference_time_ms'].mean() / quant_df['inference_time_ms'].mean():.2f}x")

    # 剪枝模型
    prune_df = df[df['pruned']]
    if not prune_df.empty:
        print(f"\n🔹 剪枝模型 ({len(prune_df)}个):")
        print(f"   平均准确率: {prune_df['accuracy'].mean():.2f}%")
        print(f"   平均大小: {prune_df['model_size_mb'].mean():.2f} MB")

    # 蒸馏模型
    distill_df = df[df['distilled']]
    if not distill_df.empty:
        print(f"\n🔹 蒸馏模型 ({len(distill_df)}个):")
        print(f"   平均准确率: {distill_df['accuracy'].mean():.2f}%")
        print(f"   平均大小: {distill_df['model_size_mb'].mean():.2f} MB")
        print(f"   平均速度: {distill_df['inference_time_ms'].mean():.2f} ms/batch")

    # 找出最佳模型
    print("\n" + "=" * 80)
    print("🏆 最佳模型")
    print("=" * 80)

    best_acc = df.loc[df['accuracy'].idxmax()]
    smallest = df.loc[df['model_size_mb'].idxmin()]
    fastest = df.loc[df['inference_time_ms'].idxmin()]

    # 计算效率分数（准确率 / 大小）
    df['efficiency_score'] = df['accuracy'] / df['model_size_mb']
    most_efficient = df.loc[df['efficiency_score'].idxmax()]

    print(f"🥇 最高准确率: {best_acc['model_name']}")
    print(f"   准确率: {best_acc['accuracy']:.2f}%")
    print(f"   大小: {best_acc['model_size_mb']:.2f} MB")

    print(f"\n🥈 最小模型: {smallest['model_name']}")
    print(f"   大小: {smallest['model_size_mb']:.2f} MB")
    print(f"   准确率: {smallest['accuracy']:.2f}%")

    print(f"\n🥉 最快推理: {fastest['model_name']}")
    print(f"   速度: {fastest['inference_time_ms']:.2f} ms/batch")
    print(f"   准确率: {fastest['accuracy']:.2f}%")

    print(f"\n💡 最高效率: {most_efficient['model_name']}")
    print(f"   效率分数: {most_efficient['efficiency_score']:.2f}")
    print(f"   准确率: {most_efficient['accuracy']:.2f}%")
    print(f"   大小: {most_efficient['model_size_mb']:.2f} MB")

    # 保存详细报告
    csv_path = str(config.RESULTS_DIR / 'comprehensive_evaluation.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细报告已保存至: {csv_path}")


def compare_optimization_effects():
    """对比优化效果（基础 vs 量化 vs 剪枝 vs 蒸馏）"""

    models_dir = str(config.MODELS_DIR)

    print("\n" + "=" * 70)
    print("🔬 优化效果对比分析")
    print("=" * 70)

    # 查找配对的模型（基础模型和它的优化版本）
    base_models = {}

    for f in os.listdir(models_dir):
        if f.startswith('best_model_') and f.endswith('.pth') and 'quantized' not in f and 'pruned' not in f:
            model_type = f.replace('best_model_', '').replace('.pth', '')
            base_models[model_type] = {
                'base': f,
                'quantized': f'best_model_{model_type}_quantized.pth' if os.path.exists(
                    os.path.join(models_dir, f'best_model_{model_type}_quantized.pth')) else None,
                'pruned': None  # 可以扩展
            }

    comparison_results = []

    for model_type, files in base_models.items():
        print(f"\n分析模型: {model_type}")
        print("-" * 70)

        results = {}

        # 评估基础模型
        if files['base']:
            base_path = os.path.join(models_dir, files['base'])
            base_result = evaluate_model(base_path, model_type, save_results=False)
            results['base'] = base_result
            print(f"  基础模型: {base_result['accuracy']:.2f}% | {base_result['model_size_mb']:.2f} MB")

        # 评估量化模型
        if files['quantized']:
            quant_path = os.path.join(models_dir, files['quantized'])
            quant_result = evaluate_model(quant_path, model_type, save_results=False)
            results['quantized'] = quant_result
            print(f"  量化模型: {quant_result['accuracy']:.2f}% | {quant_result['model_size_mb']:.2f} MB")

            # 计算变化
            if 'base' in results:
                acc_change = quant_result['accuracy'] - results['base']['accuracy']
                size_ratio = results['base']['model_size_mb'] / quant_result['model_size_mb']
                speed_ratio = results['base']['inference_time_ms'] / quant_result['inference_time_ms']

                print(f"  📊 优化效果:")
                print(f"     准确率变化: {acc_change:+.2f}%")
                print(f"     大小压缩: {size_ratio:.2f}x")
                print(f"     速度提升: {speed_ratio:.2f}x")

        comparison_results.append({
            'model_type': model_type,
            'results': results
        })

    print("\n" + "=" * 70)
    print("✅ 对比分析完成")
    print("=" * 70)


def predict_single_image(model_path, model_type, image_path, device='auto'):
    """
    对单张图片进行预测

    Args:
        model_path: 模型权重路径
        model_type: 模型类型
        image_path: 图片路径
        device: 设备
    """
    from PIL import Image

    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN,
                           std=config.IMAGENET_STD)
    ])

    # 加载模型
    model, model_info = load_model_for_evaluation(model_path, model_type, device)

    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

    # 类别名称
    classes = config.CLASSES

    print("\n" + "=" * 70)
    print("🖼️  单张图片预测结果")
    print("=" * 70)
    print(f"图片路径: {image_path}")
    print(f"模型: {os.path.basename(model_path)}")
    print(f"预测类别: {classes[predicted.item()]}")
    print(f"置信度: {confidence.item()*100:.2f}%")
    print(f"推理时间: {inference_time:.2f} ms")
    print("\n所有类别概率:")
    for i, class_name in enumerate(classes):
        prob = probabilities[0][i].item()*100
        bar = '█' * int(prob / 2)
        print(f"  {class_name:10s}: {prob:5.2f}% {bar}")
    print("=" * 70)

    return classes[predicted.item()], confidence.item()


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("📊 模型评估工具（增强版）")
    print("=" * 70)
    print("\n可用功能:")
    print("  [1] 评估单个模型")
    print("  [2] 评估所有模型（包括优化后的）")
    print("  [3] 对比优化效果（基础 vs 量化）")
    print("  [4] 预测单张图片")

    choice = input("\n请选择功能 (1-4): ").strip()

    if choice == '1':
        # 评估单个模型
        models_dir = str(config.MODELS_DIR)
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

            print("\n可用模型:")
            for i, model in enumerate(model_files, 1):
                size_mb = os.path.getsize(os.path.join(models_dir, model)) / (1024 * 1024)
                print(f"  [{i}] {model} ({size_mb:.2f} MB)")

            model_idx = int(input("\n选择模型编号: ")) - 1
            model_path = os.path.join(models_dir, model_files[model_idx])

            # 推断模型类型
            if 'distilled' in model_files[model_idx]:
                parts = model_files[model_idx].replace('distilled_', '').replace('.pth', '').split('_from_')
                model_type = parts[0]
            elif 'quantized' in model_files[model_idx]:
                model_type = model_files[model_idx].replace('best_model_', '').replace('_quantized.pth', '')
            elif 'pruned' in model_files[model_idx]:
                model_type = model_files[model_idx].replace('best_model_', '').split('_pruned_')[0]
            else:
                model_type = model_files[model_idx].replace('best_model_', '').replace('final_model_', '').replace('.pth', '')

            evaluate_model(model_path, model_type)
        else:
            print("❌ models 文件夹不存在")

    elif choice == '2':
        # 评估所有模型
        evaluate_all_models()

    elif choice == '3':
        # 对比优化效果
        compare_optimization_effects()

    elif choice == '4':
        # 预测单张图片
        models_dir = str(config.MODELS_DIR)
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

            print("\n可用模型:")
            for i, model in enumerate(model_files, 1):
                print(f"  [{i}] {model}")

            model_idx = int(input("\n选择模型编号: ")) - 1
            model_path = os.path.join(models_dir, model_files[model_idx])

            # 推断模型类型
            if 'distilled' in model_files[model_idx]:
                parts = model_files[model_idx].replace('distilled_', '').replace('.pth', '').split('_from_')
                model_type = parts[0]
            else:
                model_type = model_files[model_idx].replace('best_model_', '').replace('_quantized.pth', '').replace('.pth', '')

            image_path = input("输入图片路径: ").strip()
            predict_single_image(model_path, model_type, image_path)
        else:
            print("❌ models 文件夹不存在")

    print("\n" + "=" * 70)
    print("✅ 评估完成！")
    print("=" * 70)