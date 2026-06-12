# 全面审查报告：八维并行多 Agent 审查

日期：2026-06-12
分支：`feat/comprehensive-optimization`（PR #1）
修复提交：`7671312`
方法：8 个独立审查 agent 并行，各持一份维度检查清单，主会话交叉核实后统一修复。
验证：71 测试全绿（审查前 59，新增 12 回归测试），`ruff check` 干净。

## 结论

**0 critical / 13 high（已全部修复）/ 若干 medium·low（见遗留清单）**

## 已修复的 high 级问题

### 优化链路
- 量化 checkpoint 在有 GPU 的机器上加载即崩（qint8 无法映射 CUDA）→ `inference.py` 检出 `quantized` 标记后强制 CPU。
- 蒸馏时 teacher 被喂 student 尺寸输入（如 resnet50 吃 48×48），软标签严重失真 → 师生尺寸不同时对 teacher 输入做双线性插值（`optimize_distill.py`）。

### 边缘推理核心
- `EmotionSmoother` 滞回卡死：锁定类别概率坍缩后永远输出旧标签（表情过渡期常态触发）→ 锁定类 EMA 跌破 `threshold - hysteresis` 自动释放。
- 无人脸时不重置状态，换人入镜被前一人的 EMA 污染 → `note_no_face()` 连续 15 帧无脸整体重置（`pipeline.py` 上报）。
- `assets.ensure` 下载中断留下半截文件，下次启动误判已缓存，报与下载无关的 protobuf 错误 → 临时文件 `.part` + `os.replace` 原子改名。

### Web / 安全
- `/api/load_model` 路径穿越：`filename` 未校验，`..\\` 或绝对路径可让服务对任意文件执行 `torch.load` → 纯文件名 + `.pth` 后缀校验。
- `torch.load` 两处未显式 `weights_only=True`（行为随 torch 版本漂移，旧版为任意 pickle 反序列化 RCE 面）→ 显式声明。
- 推理在持锁状态下进行，所有 `/api/predict` 请求被串行化 → 锁内仅做引用快照，前向在锁外。
- `/api/models` 每次刷新对每个 `.pth` 全量反序列化权重只为读 5 个标量 → 按 `(mtime, size)` 缓存元信息。
- 前端仅 `#uploadBox` 阻止拖拽默认行为，落点稍偏浏览器直接打开文件 → window 级 `dragover/drop` 防护。
- README 引导 `FLASK_HOST=0.0.0.0 FLASK_DEBUG=1` 危险组合（Werkzeug 调试器 RCE）→ 代码层 debug 强制本机监听，README 拆开示例并加警告。
- 无解压炸弹防护，10MB 高压缩比图片可解出数亿像素 → `Image.MAX_IMAGE_PIXELS = 24_000_000`。
- 错误响应回传 `str(e)` 泄漏内部路径 → 对外统一文案，详情仅入服务端日志。

### 文档
- README 声明 MIT 但 LICENSE 不存在 → 已补（后与远端 main 用户自加版本合并，署名以远端为准）。
- 结构树缺失 `src/assets.py`、`src/{face,engine,runtime}/`、`tests/`、`docs/` → 补全。
- `/api/load_model` 示例含已废弃的 `model_type` 字段、学习率调度描述与实现不符（监控的是准确率非损失）→ 修正。

### 顺带修复（medium）
- CUDA 基准计时未 `synchronize`，GPU 数字严重偏小 → 前后同步（`evaluate.py`）。
- 前端按原始文件 10MB 校验但 base64 膨胀 4/3，7.5~10MB 图片通过前端却被服务端 413 → 前端按 3/4 折算并更新文案。
- `run_camera` 异常时摄像头句柄泄漏 → try/finally。
- `demo.build` 反向校验：未给 `--emotion-model` 却给 `--labels`/`--input-size` 时直接报错（原本静默组合出错误标签映射）。
- 默认模型加载失败不回退下一个 → `break` 移入成功路径。
- `EmotionEvent.probs` 持有 smoother 内部 `_ema` 引用 → 返回副本。

## 新增回归测试（12 个）

- 路径穿越/坏后缀拒绝、缺 filename 400、不存在 404、坏 base64 400（`test_app.py`）
- 平滑器锁定释放、连续无脸重置、EMA 返回副本（`test_smoother.py`）
- 下载失败不留残缺文件（`test_assets.py`）
- `TrainConfig` ↔ `train_model` 签名契约（`test_train_contract.py`）

## 遗留清单（medium / low，未修，按维度）

### 训练链路（全部 low）
- 验证阶段未启用 autocast（仅速度影响）；DataLoader 未传 `generator`/`worker_init_fn`，num_workers>0 时增强随机性不可复现；未设 `persistent_workers`（Windows spawn 每 epoch 重建 worker）；epoch loss 为 batch 均值的均值；Adam+weight_decay 可换 AdamW；自定义 CNN 用 ImageNet 统计量归一化。

### 优化链路
- [low] 量化压缩比把含 optimizer 状态的训练 checkpoint 当分母，倍数被高估。
- [low] `torch.quantization.*` 已弃用，应迁 `torch.ao.quantization`。
- [low] 剪枝后无微调入口；非结构化稀疏在 dense 推理上无速度收益。

### 边缘核心
- [medium→建议 Phase 2] 资产 URL 指向 `raw/main` 可变引用，无 sha256 校验——建议 pin commit hash + `Asset.sha256` 字段（需联网核实哈希，故未在本轮落地）。
- [low] 脸贴近画面边缘时非对称钳制导致裁剪拉伸畸变（可 copyMakeBorder 补边）；crop 与 preprocess 双重 resize 冗余；`OnnxRuntimeEngine` 可用输出维度校验 `len(labels)` 作最后防线。

### Web
- [low] `/api/status` 读字段不经锁（CPython 下窗口极小）；debug reloader 下默认模型加载两遍。

### 测试
- [medium] `test_engine` 每测试重复导出 ONNX、`test_app` 重复建 checkpoint，可抽 module fixture；`crop_face` 全越界框、`/api/models` 端点无覆盖。
- [low] `test_model` 与 `test_model_input_size` 前向遍历重复；dataset 清洗路径覆盖薄；蒸馏损失仅验 alpha=1 端点；随机输入未固定 seed。

### 工程卫生
- [medium] checkpoint 目录遍历三处重复（evaluate/app×2）；蒸馏 epoch 循环与 `train._run_epoch` 高度同构。
- [low] 死代码：`get_class_name`、`auto_clean` 构造参数（及 `clean_directory` 的 `__new__` hack）、`eval_mode` 参数；运行时依赖未按场景分组（边缘链路只需 cv2+ort+numpy，可拆 extras）；`.ruff_cache/` 未入 .gitignore。

### 文档
- [low] `/api/predict` 响应缺 `en` 字段示例、multipart 方式未文档化；`/api/models`、`/api/load_model` 响应体未文档化；plan 文档中 `fetch_yunet.py` 引用已由备注说明（见各文档头部）。

## 审查方法备注

每个维度 agent 持只读权限独立审查，输出 `[严重度][文件:行号][问题][建议]` 结构化发现；主会话逐条对照源码核实后才动手修复——本轮 13 条 high 全部与代码吻合，无误报。
