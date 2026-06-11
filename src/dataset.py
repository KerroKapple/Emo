"""dataset.py - 表情识别数据集（含数据清洗与去重）"""

import os
import shutil
import hashlib
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import src.config as config
from src.logging_setup import get_logger

logger = get_logger('emotion.dataset')

_VALID_FORMATS = {'JPEG', 'PNG', 'BMP'}
_VALID_MODES = {'RGB', 'L', 'RGBA'}


class EmotionDataset(Dataset):
    """按类别子目录组织的表情数据集，支持加载时校验与首次自动清洗"""

    def __init__(self, root_dir, transform=None, auto_clean=False, clean_on_load=False):
        self.root_dir = root_dir
        self.transform = transform
        self.clean_on_load = clean_on_load
        self.classes = config.CLASSES
        self.class_to_idx = config.CLASS_TO_IDX
        self.clean_stats = _new_stats()

        if auto_clean:
            self._auto_clean_dataset()

        self.samples = []
        self._load_samples()
        logger.info("数据集加载完成: %d 张有效图片", len(self.samples))

    def _check_image_valid(self, img_path):
        """返回 (是否有效, 错误类型)"""
        try:
            with Image.open(img_path) as img:
                if img.format not in _VALID_FORMATS:
                    return False, 'invalid_format'
                w, h = img.size
                if w < 32 or h < 32 or w > 2000 or h > 2000:
                    return False, 'size_error'
                if img.mode not in _VALID_MODES:
                    return False, 'invalid_format'
                arr = np.array(img.convert('RGB'))
                if arr.mean() < 5 or arr.mean() > 250 or arr.var() < 10:
                    return False, 'quality_error'
            return True, None
        except Exception:
            return False, 'corrupted'

    @staticmethod
    def _file_hash(file_path):
        h = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    def _auto_clean_dataset(self):
        """移动损坏/异常/重复图片到隔离区"""
        quarantine = str(config.QUARANTINE_DIR)
        os.makedirs(quarantine, exist_ok=True)
        logger.info("开始自动清洗: %s", self.root_dir)

        problems = []
        hashes = defaultdict(list)
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(config.IMAGE_EXTENSIONS)]
            for filename in tqdm(files, desc=f"扫描 {class_name}", ncols=80):
                path = os.path.join(class_dir, filename)
                self.clean_stats['total_files'] += 1
                valid, err = self._check_image_valid(path)
                if not valid:
                    problems.append((path, err))
                    self.clean_stats[err] += 1
                else:
                    fh = self._file_hash(path)
                    if fh:
                        hashes[fh].append(path)

        for path, err in problems:
            self._quarantine(path, os.path.join(quarantine, err))

        duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
        dup_count = sum(len(v) - 1 for v in duplicates.values())
        for files in duplicates.values():
            for path in files[1:]:
                self._quarantine(path, os.path.join(quarantine, 'duplicates'))

        logger.info("清洗完成: 问题 %d / 重复 %d -> %s", len(problems), dup_count, quarantine)

    def _quarantine(self, path, dst_root):
        rel = os.path.relpath(path, self.root_dir)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.move(path, dst)
        except OSError:
            pass

    def _load_samples(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning("类别目录不存在: %s", class_dir)
                continue
            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(config.IMAGE_EXTENSIONS)]
            for img_name in files:
                path = os.path.join(class_dir, img_name)
                if self.clean_on_load:
                    valid, err = self._check_image_valid(path)
                    if not valid:
                        if err:
                            self.clean_stats[err] += 1
                        continue
                self.samples.append((path, self.class_to_idx[class_name]))
                self.clean_stats['valid_files'] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning("读取失败 %s: %s", img_path, e)
            image = Image.new('RGB', (config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE))
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_name(self, idx):
        return self.classes[idx]

    def get_class_distribution(self):
        dist = {c: 0 for c in self.classes}
        for _, label in self.samples:
            dist[self.classes[label]] += 1
        return dist

    @staticmethod
    def clean_directory(data_dir):
        """独立清洗某目录，返回清洗统计"""
        temp = EmotionDataset.__new__(EmotionDataset)
        temp.root_dir = data_dir
        temp.classes = config.CLASSES
        temp.clean_stats = _new_stats()
        temp._auto_clean_dataset()
        return temp.clean_stats


def _new_stats():
    return {'total_files': 0, 'valid_files': 0, 'corrupted': 0,
            'invalid_format': 0, 'size_error': 0, 'quality_error': 0}


def quick_clean(data_dir):
    """快速清洗某目录（不构建数据集）"""
    return EmotionDataset.clean_directory(data_dir)


def _parse_args():
    p = argparse.ArgumentParser(description='数据集加载与清洗')
    p.add_argument('--root', default=str(config.RAW_DIR), help='数据根目录')
    p.add_argument('--clean', action='store_true', help='仅执行清洗')
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.clean:
        stats = quick_clean(args.root)
        logger.info("清洗统计: %s", stats)
    else:
        ds = EmotionDataset(args.root, clean_on_load=False)
        logger.info("类别分布: %s", ds.get_class_distribution())
