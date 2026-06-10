# P1 工程地基 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把表情识别项目重构为可配置、路径稳健、无重复、有日志和测试的工程地基，为后续 P2（模型效果）/P3（部署）/P4（Web）打底。

**Architecture:** 引入单一配置模块 `src/config.py`（集中常量、路径、超参）和 `src/logging_setup.py`（统一日志）；把 `model.py` 里 5 个几乎重复的迁移学习包装类收敛为「注册表 + 单一 TransferModel」；所有脚本改用 `config` 里的绝对路径常量（基于 `pathlib` 从项目根解析，不再依赖 cwd）；用 uv + pyproject 管理依赖并加 pytest 测试。

**Tech Stack:** Python 3.10+、uv、PyTorch / torchvision、Flask、pytest。

**重要前提（实施者必读）:**
- 本项目全局规则：**只用 uv**（`uv run` / `uv add`），禁止 system python 和 `uv pip`，禁止手改 `pyproject.toml`。
- **无向后兼容**：`model.py` 重构后，旧 `.pth` checkpoint 的 `state_dict` 键名会从 `resnet.*/vgg.*/mobilenet.*/efficientnet.*` 变为 `backbone.*`。旧权重将无法直接加载——这是预期且可接受的（权重已 gitignore，P2 会重新训练）。
- 代码与注释用**中文**，注释从简。

---

## File Structure

**新建:**
- `pyproject.toml` — uv 管理的依赖与项目元数据（由 `uv init` / `uv add` 生成，不手改）
- `src/config.py` — 集中配置：项目根、各数据/模型/结果目录、类别常量、中文名/emoji、ImageNet 归一化常量、图像尺寸、`TrainConfig` 数据类
- `src/logging_setup.py` — `get_logger(name)` 统一日志配置
- `tests/conftest.py` — pytest 共享 fixture（项目根、临时数据目录）
- `tests/test_config.py` — 配置模块测试
- `tests/test_logging_setup.py` — 日志模块测试
- `tests/test_model.py` — 模型工厂与各架构前向输出形状测试
- `tests/test_dataset.py` — 数据集类别常量与默认行为测试

**修改:**
- `src/model.py` — 收敛重复包装类为注册表 + `TransferModel`，改用 torchvision `weights=` 新 API
- `src/dataset.py` — 类别常量改从 `config` 导入；`clean_on_load` 默认改为 `False`；隔离区路径用 `config`
- `src/utils.py` — `split_dataset` 类别常量改从 `config` 导入；默认路径参数用 `config`
- `src/train.py` — 用 `TrainConfig` + `config` 路径常量 + 日志
- `src/optimize_distill.py` — 路径常量与归一化常量改用 `config`
- `src/train_multiple.py` — 结果/进度图路径改用 `config`
- `app.py` — 类别/中文名/emoji/预处理改从 `config` 导入；`debug` 改由环境变量控制；模型目录用 `config`

**删除:**
- `requirements.txt`、`requirements_web.txt` — 依赖统一到 `pyproject.toml`（无向后兼容）

---

### Task 0: uv 项目初始化与依赖

**Files:**
- Create: `pyproject.toml`（由 uv 生成）
- Delete: `requirements.txt`, `requirements_web.txt`

- [ ] **Step 1: 初始化 uv 项目**

Run:
```bash
cd E:/Emo && uv init --bare --name emotion-recognition --python 3.10
```
Expected: 生成 `pyproject.toml`（`--bare` 不生成示例 `hello.py`）。若提示已存在则跳过。

- [ ] **Step 2: 添加运行期依赖**

Run:
```bash
cd E:/Emo && uv add torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm flask flask-cors
```
Expected: `pyproject.toml` 出现 `[project] dependencies`，生成 `uv.lock` 与 `.venv`。

- [ ] **Step 3: 添加测试依赖（dev 组）**

Run:
```bash
cd E:/Emo && uv add --dev pytest
```
Expected: `pytest` 进入 dev 依赖。

- [ ] **Step 4: 验证环境可用**

Run:
```bash
cd E:/Emo && uv run python -c "import torch, torchvision, flask; print('ok')"
```
Expected: 输出 `ok`。

- [ ] **Step 5: 删除旧依赖文件并提交**

```bash
cd E:/Emo && git rm requirements.txt requirements_web.txt
git add pyproject.toml uv.lock
git commit -m "chore: 用 uv/pyproject 统一依赖管理，移除 requirements"
```

---

### Task 1: 集中配置模块 `src/config.py`

**Files:**
- Create: `src/config.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
from pathlib import Path
import src.config as config


def test_classes_and_index():
    assert config.CLASSES == ['anger', 'fear', 'happy', 'sad', 'surprise']
    assert config.NUM_CLASSES == 5
    assert config.CLASS_TO_IDX['happy'] == 2


def test_zh_and_emoji_cover_all_classes():
    for cls in config.CLASSES:
        assert cls in config.CLASS_NAMES_ZH
        assert cls in config.CLASS_EMOJIS


def test_paths_are_under_project_root():
    root = config.PROJECT_ROOT
    assert isinstance(root, Path)
    assert config.RAW_DIR == root / 'data' / 'raw'
    assert config.TRAIN_DIR == root / 'data' / 'train'
    assert config.VAL_DIR == root / 'data' / 'val'
    assert config.MODELS_DIR == root / 'models'
    assert config.RESULTS_DIR == root / 'results'


def test_train_config_defaults():
    cfg = config.TrainConfig()
    assert cfg.model_type == 'resnet18'
    assert cfg.num_epochs == 20
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.001
    assert cfg.device == 'auto'
```

- [ ] **Step 2: 创建 `tests/conftest.py` 让 `src` 可被导入**

```python
import sys
from pathlib import Path

# 把项目根加入 sys.path，使 `import src.xxx` 在测试中可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd E:/Emo && uv run pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.config'`）

- [ ] **Step 4: 实现 `src/config.py`**

```python
"""config.py - 集中配置：路径、类别常量、超参"""

from dataclasses import dataclass
from pathlib import Path

# 项目根目录：本文件位于 <root>/src/config.py，向上两级即根
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据与产物目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
QUARANTINE_DIR = DATA_DIR / 'quarantine'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 类别
CLASSES = ['anger', 'fear', 'happy', 'sad', 'surprise']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

CLASS_NAMES_ZH = {
    'anger': '愤怒',
    'fear': '恐惧',
    'happy': '快乐',
    'sad': '悲伤',
    'surprise': '惊讶',
}

CLASS_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
}

# 图像预处理常量（ImageNet 归一化）
# 注意：IMAGE_SIZE 在 P1 保持 48 以不改变现有行为；48→224 的修正属于 P2
IMAGE_SIZE = 48
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 支持的图片扩展名
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


@dataclass
class TrainConfig:
    """训练超参配置"""
    model_type: str = 'resnet18'
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    device: str = 'auto'  # 'auto' | 'cuda' | 'cpu'
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd E:/Emo && uv run pytest tests/test_config.py -v`
Expected: PASS（4 项全过）

- [ ] **Step 6: 提交**

```bash
cd E:/Emo && git add src/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: 新增集中配置模块 config.py"
```

---

### Task 2: 统一日志模块 `src/logging_setup.py`

**Files:**
- Create: `src/logging_setup.py`
- Test: `tests/test_logging_setup.py`

- [ ] **Step 1: 写失败测试 `tests/test_logging_setup.py`**

```python
import logging
from src.logging_setup import get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger('emotion.test')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'emotion.test'


def test_get_logger_has_single_handler():
    logger = get_logger('emotion.handlers')
    # 多次获取不应重复挂 handler
    logger2 = get_logger('emotion.handlers')
    assert len(logger2.handlers) == 1
    assert logger is logger2


def test_logger_level_is_info():
    logger = get_logger('emotion.level')
    assert logger.level == logging.INFO
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd E:/Emo && uv run pytest tests/test_logging_setup.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.logging_setup'`）

- [ ] **Step 3: 实现 `src/logging_setup.py`**

```python
"""logging_setup.py - 统一日志配置"""

import logging

_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_DATEFMT = '%H:%M:%S'


def get_logger(name: str) -> logging.Logger:
    """返回配置好的命名 logger；重复调用同名不会重复挂 handler"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd E:/Emo && uv run pytest tests/test_logging_setup.py -v`
Expected: PASS（3 项全过）

- [ ] **Step 5: 提交**

```bash
cd E:/Emo && git add src/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: 新增统一日志模块 logging_setup.py"
```

---

### Task 3: 收敛 `src/model.py` 重复包装类

**Files:**
- Modify: `src/model.py`（替换 `EmotionResNet`/`EmotionVGG`/`EmotionMobileNet`/`EmotionEfficientNet` 四个类与 `get_model`）
- Test: `tests/test_model.py`

- [ ] **Step 1: 写失败测试 `tests/test_model.py`**

```python
import torch
import pytest
from src.model import get_model

ALL_TYPES = ['cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet', 'efficientnet']


@pytest.mark.parametrize('model_type', ALL_TYPES)
def test_forward_output_shape(model_type):
    model = get_model(model_type, num_classes=5, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 5)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        get_model('not_a_model')
```

- [ ] **Step 2: 运行测试确认失败（基线，可能因旧 API 报弃用或仍通过）**

Run: `cd E:/Emo && uv run pytest tests/test_model.py -v`
Expected: 旧实现下应能 PASS（这是重构前的特征化测试基线）。记录为通过；若因 `pretrained=` 弃用报错则视为 FAIL，按 Step 3 修复。

- [ ] **Step 3: 替换 `src/model.py` 中四个迁移学习类与 `get_model`**

把 `EmotionResNet`、`EmotionVGG`、`EmotionMobileNet`、`EmotionEfficientNet` 四个类（约第 99–359 行）整体删除，替换为以下注册表与单一包装类（`EmotionCNN` 保留不动）：

```python
from torchvision import models

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
```

然后把 `get_model` 函数体替换为：

```python
def get_model(model_type='cnn', num_classes=5, pretrained=True):
    """模型工厂：'cnn' 用自定义 CNN，其余走迁移学习注册表"""
    if model_type == 'cnn':
        return EmotionCNN(num_classes=num_classes)
    if model_type in _TRANSFER_REGISTRY:
        return TransferModel(model_type, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"不支持的模型类型: {model_type}")
```

注意：删除文件顶部各旧类内部重复的 `from torchvision import models` 局部导入；统一在 `_TRANSFER_REGISTRY` 上方做一次模块级导入。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd E:/Emo && uv run pytest tests/test_model.py -v`
Expected: PASS（7 个参数化 + 1 个异常 = 8 项全过，且无 `pretrained` 弃用告警）

- [ ] **Step 5: 提交**

```bash
cd E:/Emo && git add src/model.py tests/test_model.py
git commit -m "refactor: 收敛模型包装为注册表+TransferModel，改用 weights API"
```

---

### Task 4: 重构 `src/dataset.py`

**Files:**
- Modify: `src/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: 写失败测试 `tests/test_dataset.py`**

```python
import inspect
from src.dataset import EmotionDataset
import src.config as config


def test_dataset_uses_config_classes():
    # 通过 __init__ 默认构造一个空目录数据集，验证类别来自 config
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        for cls in config.CLASSES:
            os.makedirs(os.path.join(d, cls))
        ds = EmotionDataset(d, clean_on_load=False)
        assert ds.classes == config.CLASSES
        assert ds.class_to_idx == config.CLASS_TO_IDX


def test_clean_on_load_default_is_false():
    sig = inspect.signature(EmotionDataset.__init__)
    assert sig.parameters['clean_on_load'].default is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd E:/Emo && uv run pytest tests/test_dataset.py -v`
Expected: FAIL（`test_clean_on_load_default_is_false` 失败，因当前默认是 `True`）

- [ ] **Step 3: 修改 `src/dataset.py`**

在文件顶部 import 区加入：
```python
import src.config as config
```

将 `__init__` 签名（第 20 行）的默认值改为：
```python
    def __init__(self, root_dir, transform=None, auto_clean=False, clean_on_load=False):
```

将第 35–36 行类别定义改为引用 config：
```python
        self.classes = config.CLASSES
        self.class_to_idx = config.CLASS_TO_IDX
```

将 `_auto_clean_dataset` 内隔离区路径（第 125 行）改为：
```python
        quarantine_dir = str(config.QUARANTINE_DIR)
        os.makedirs(quarantine_dir, exist_ok=True)
```

将静态方法 `clean_directory` 内重复的类别列表（第 334 行）改为：
```python
        temp_dataset.classes = config.CLASSES
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd E:/Emo && uv run pytest tests/test_dataset.py -v`
Expected: PASS（2 项全过）

- [ ] **Step 5: 提交**

```bash
cd E:/Emo && git add src/dataset.py tests/test_dataset.py
git commit -m "refactor: dataset 类别常量统一来自 config，clean_on_load 默认关闭"
```

---

### Task 5: 重构 `src/utils.py`

**Files:**
- Modify: `src/utils.py`

- [ ] **Step 1: 修改 import 与 `split_dataset`**

在顶部 import 区加入：
```python
import src.config as config
```

将 `split_dataset` 内重复类别列表（第 44 行）改为：
```python
    classes = config.CLASSES
```

将其 `__main__` 测试块（第 242–247 行）中的相对路径改为：
```python
    raw_dir = str(config.RAW_DIR)
    train_dir = str(config.TRAIN_DIR)
    val_dir = str(config.VAL_DIR)
```

- [ ] **Step 2: 冒烟验证模块可导入**

Run: `cd E:/Emo && uv run python -c "import src.utils; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
cd E:/Emo && git add src/utils.py
git commit -m "refactor: utils 类别与路径统一走 config"
```

---

### Task 6: 重构 `src/train.py`

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: 替换 import 区与日志**

把顶部 import 区改为（新增 config / logging / dataclasses）：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
from dataclasses import asdict

import src.config as config
from src.logging_setup import get_logger
from src.dataset import EmotionDataset
from src.model import get_model, count_parameters
from src.utils import plot_training_history, save_checkpoint

logger = get_logger('emotion.train')
```

- [ ] **Step 2: 用 config 常量替换 `train_model` 内硬编码**

将归一化常量（第 144–145、151–152 行）改为 `config.IMAGENET_MEAN` / `config.IMAGENET_STD`，`Resize` 改为 `(config.IMAGE_SIZE, config.IMAGE_SIZE)`。

将数据路径（第 157–158 行）改为：
```python
    train_dataset = EmotionDataset(str(config.TRAIN_DIR), transform=train_transform)
    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=val_transform)
```

将创建模型时的类别数（第 178 行）改为 `num_classes=config.NUM_CLASSES`。

将保存路径（第 239、254、262 行）改为：
```python
        save_path = str(config.MODELS_DIR / f'best_model_{model_type}.pth')
        ...
    final_path = str(config.MODELS_DIR / f'final_model_{model_type}.pth')
        ...
    plot_training_history(history, str(config.RESULTS_DIR / f'training_history_{model_type}.png'))
```

- [ ] **Step 3: 用 `TrainConfig` 替换 `__main__` 硬编码 config**

把第 268–294 行的 `config = {...}` 字典替换为：
```python
    train_cfg = config.TrainConfig(model_type='efficientnet')
    logger.info("训练配置: %s", asdict(train_cfg))
    model, history = train_model(**asdict(train_cfg))
```

- [ ] **Step 4: 冒烟验证模块可导入**

Run: `cd E:/Emo && uv run python -c "import src.train; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 5: 提交**

```bash
cd E:/Emo && git add src/train.py
git commit -m "refactor: train 使用 TrainConfig 与 config 路径/常量"
```

---

### Task 7: 重构 `src/optimize_distill.py`

**Files:**
- Modify: `src/optimize_distill.py`

- [ ] **Step 1: 替换 import 区**

把顶部 import 区的：
```python
from dataset import EmotionDataset
from model import get_model
from utils import save_checkpoint
```
改为：
```python
import src.config as config
from src.dataset import EmotionDataset
from src.model import get_model
from src.utils import save_checkpoint
```

- [ ] **Step 2: 用 config 常量替换硬编码路径/常量**

将 `knowledge_distillation` 内归一化常量（第 273–274、280–281 行）改为 `config.IMAGENET_MEAN` / `config.IMAGENET_STD`，`Resize` 改为 `(config.IMAGE_SIZE, config.IMAGE_SIZE)`。

将数据路径（第 286–287 行）改为：
```python
    train_dataset = EmotionDataset(str(config.TRAIN_DIR), transform=train_transform)
    val_dataset = EmotionDataset(str(config.VAL_DIR), transform=val_transform)
```

将学生模型保存路径（第 443 行）改为：
```python
            save_path = str(config.MODELS_DIR / f'distilled_{student_model_type}_from_{teacher_model_type}.pth')
```

将 `__main__` 内 `models_dir = '../models'`（第 482、499、557 行）统一改为：
```python
        models_dir = str(config.MODELS_DIR)
```

将教师模型路径（第 539 行）改为：
```python
        teacher_path = str(config.MODELS_DIR / f'best_model_{teacher_type}.pth')
```

将 `num_classes=5` 出现处改为 `config.NUM_CLASSES`。

- [ ] **Step 3: 冒烟验证模块可导入**

Run: `cd E:/Emo && uv run python -c "import src.optimize_distill; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 4: 提交**

```bash
cd E:/Emo && git add src/optimize_distill.py
git commit -m "refactor: optimize_distill 路径与常量统一走 config"
```

---

### Task 8: 重构 `src/train_multiple.py`

**Files:**
- Modify: `src/train_multiple.py`

- [ ] **Step 1: 替换 import 区**

把 `from train import train_model` 改为：
```python
import src.config as config
from src.train import train_model
```

- [ ] **Step 2: 用 config 路径替换结果/进度图保存路径**

将以下硬编码路径全部改用 `config.RESULTS_DIR`：
- 第 207 行 `save_path = '../results/training_progress_realtime.png'` →
  ```python
      save_path = str(config.RESULTS_DIR / 'training_progress_realtime.png')
  ```
- 第 329 行 `csv_path = '../results/models_comparison.csv'` →
  ```python
      csv_path = str(config.RESULTS_DIR / 'models_comparison.csv')
  ```
- 第 443 行 `save_path = '../results/models_comparison.png'` →
  ```python
      save_path = str(config.RESULTS_DIR / 'models_comparison.png')
  ```

- [ ] **Step 3: 冒烟验证模块可导入**

Run: `cd E:/Emo && uv run python -c "import src.train_multiple; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 4: 提交**

```bash
cd E:/Emo && git add src/train_multiple.py
git commit -m "refactor: train_multiple 结果路径统一走 config"
```

---

### Task 9: 重构 `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 替换 import 与全局常量**

把第 14–46 行的 `sys.path` 注入、`from src.model import get_model`、`classes`/`class_names_zh`/`class_emojis` 三个字典全部替换为：
```python
import src.config as config
from src.model import get_model
from src.logging_setup import get_logger

logger = get_logger('emotion.app')

# 全局变量
model = None
device = None
transform = None
classes = config.CLASSES
class_names_zh = config.CLASS_NAMES_ZH
class_emojis = config.CLASS_EMOJIS
```
（删除原 `sys.path.append(...)` 一行；改用包内绝对导入 `src.model`，运行方式见 Step 5。）

- [ ] **Step 2: `load_model` 内预处理常量改用 config**

将第 77–82 行 transform 中的 `Resize((48, 48))` 改为 `(config.IMAGE_SIZE, config.IMAGE_SIZE)`，归一化改为 `config.IMAGENET_MEAN` / `config.IMAGENET_STD`，`num_classes=5` 改为 `config.NUM_CLASSES`。把第 55、84、85 行的 `print(...)` 改为 `logger.info(...)`。

- [ ] **Step 3: 模型目录改用 config**

将第 133 行 `models_dir = 'models'` 与第 177 行 `os.path.join('models', model_filename)`、第 237 行默认模型路径改为：
```python
    models_dir = str(config.MODELS_DIR)        # 第 133 行
    ...
    model_path = os.path.join(str(config.MODELS_DIR), model_filename)   # 第 177 行
    ...
    default_model = str(config.MODELS_DIR / 'best_model_resnet18.pth')  # 第 237 行
```

- [ ] **Step 4: `debug` 改由环境变量控制**

将第 257 行 `app.run(debug=True, host='0.0.0.0', port=5000)` 改为：
```python
    import os
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    app.run(debug=debug, host=host, port=port)
```

- [ ] **Step 5: 冒烟验证可导入（以模块方式，使 `src` 包可解析）**

Run: `cd E:/Emo && uv run python -c "import app; print('ok')"`
Expected: 输出 `ok`（无模型时不应抛错，仅日志提示）

- [ ] **Step 6: 提交**

```bash
cd E:/Emo && git add app.py
git commit -m "refactor: app 复用 config 常量，debug/host 改由环境变量控制"
```

---

### Task 10: 收尾 — `.gitignore` 修正、`src/__init__.py`、README、全量测试

**Files:**
- Create: `src/__init__.py`（确保 `src` 是可导入包）
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: 创建空的 `src/__init__.py`**

```bash
cd E:/Emo && uv run python -c "open('src/__init__.py','a').close()"
```

- [ ] **Step 2: 修正 `.gitignore` 大小写并补充**

把第 2 行 `Data/` 改为 `data/`，并补充 uv/pytest 产物：
```gitignore
# 忽略数据和模型（这是最关键的）
data/
models/
results/

# 忽略开发工具和缓存
.idea/
__pycache__/
*.pyc
.pytest_cache/
.venv/
```
（`uv.lock` 与 `pyproject.toml` 保留纳入版本控制，不忽略。）

- [ ] **Step 3: 更新 README 安装/运行段落**

把 README「环境配置」与各运行命令中的 `pip install -r requirements.txt`、`python xxx.py`、`cd src` 等改为 uv 形式，例如：
```bash
# 安装依赖
uv sync

# 数据划分
uv run python -m src.utils

# 训练
uv run python -m src.train

# 启动 Web（默认仅本机访问；如需对外/调试用环境变量）
uv run python app.py
```

- [ ] **Step 4: 运行全量测试确认绿**

Run: `cd E:/Emo && uv run pytest -v`
Expected: 所有测试 PASS（config 4 + logging 3 + model 8 + dataset 2 = 17 项）

- [ ] **Step 5: 提交**

```bash
cd E:/Emo && git add src/__init__.py .gitignore README.md
git commit -m "chore: 修正 gitignore、补 src 包标识、README 改用 uv"
```

---

## Self-Review

**Spec coverage（对照 P1 目标）:**
- 配置化（消除硬编码 config）→ Task 1 + Task 6 ✅
- 路径稳健（绝对路径、不依赖 cwd）→ Task 1 + Tasks 5–9 ✅
- 消除重复（类别常量、模型包装类、归一化常量）→ Tasks 3/4/5/6/7/9 ✅
- 日志 → Task 2 + Tasks 6/9 ✅
- torchvision 新 API → Task 3 ✅
- 测试 → Tasks 1/2/3/4 + Task 10 全量 ✅
- 依赖管理（uv）→ Task 0 + Task 10 ✅

**Placeholder 扫描:** 无 TBD/TODO；所有代码块为完整可粘贴内容。✅

**类型一致性:** `get_model` / `TransferModel` / `TrainConfig` / `get_logger` 签名在各任务中一致；`config.*` 常量名跨任务统一。✅

**已知影响（非缺陷，需实施者知晓）:**
- 旧 `.pth` checkpoint 键名不兼容（model.py 重构所致）——符合「无向后兼容」策略，P2 重训解决。
- `train.py`/`optimize_distill.py` 等脚本现以 `python -m src.xxx` 方式运行（因改用 `src.` 包内绝对导入），README 已同步。
