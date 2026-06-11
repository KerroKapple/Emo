# 设计：陪伴机器人「情绪感知核心」（嵌入式表情识别）

日期：2026-06-11
状态：设计稿（待评审）

## 1. 背景与定位

项目从「网页表情识别 Demo」转型为**陪伴机器人的情绪感知核心**：设备有屏幕 + 语音，新增一颗摄像头，
实时"看用户脸色"驱动机器人的表情/语音反应。

核心命题不是刷榜精度，而是 **在便宜、低功耗的边缘硬件上，做出"反应像样、跟手"的情绪识别**。
关键矛盾：用户要"准确率"又要"嵌入式"——通过 **大模型当老师追精度 → 蒸馏小学生模型出厂** 调和。

> 现状约束：开发机无数据集、无 NPU、无 GPU（见 memory）。训练/QAT 在用户的 GPU 机器进行；
> 本机可开发并验证：管线代码、ONNX 导出、INT8 PTQ 标定、YuNet 集成、以及基于 onnxruntime 的 PC 摄像头 Demo（合成帧）。

## 2. 目标与约束

- **精度目标（现实值，来自调研）**：RAF-DB 7 类，轻量模型可达 ~85–88%（EfficientFace 1.28M 参数 88.28%）；
  叠加人脸裁剪 + 时序平滑后，体感可靠度更高。AffectNet 8 类更难（SOTA ~63–67%），如实告知。
- **实时性**：单脸推理目标 < 30ms（NPU 5–10ms / CPU 20–50ms），几 FPS 足够陪伴反应。
- **模型预算**：int8，< 10MB，适配 ~1 TOPS NPU 或纯 ARM CPU。
- **硬件无关优先**：以 **ONNX (opset 13)** 为标准产物，硬件适配（RKNN/ncnn/ORT）作为可插拔的最后一步。
- **参考硬件**：瑞芯微 RK3566/3568 类（带 ~1 TOPS NPU，陪伴机器人常见、量产成熟）；纯 CPU 用 ncnn 兜底。

## 3. 关键技术决策（含调研依据）

1. **人脸检测裁剪（最大精度杠杆）**：用 **YuNet**（75,856 参数、~1.6ms、OpenCV 内置 `cv2.FaceDetectorYN`）
   先检测+裁剪+对齐人脸，再送情绪分类。训练与推理两侧都做，消除背景/尺度干扰。
2. **数据集升级**：弃用 FER-2013（48×48 灰度、噪声大、天花板低）。改用 **RAF-DB**（100×100 RGB、~15k、7 类，干净）
   作主力；可选 **AffectNet**（~29 万、8 类）做更强老师。
3. **类别集合**：现 5 类缺 **neutral**——陪伴场景里用户大多数时间是中性脸，**必须加 neutral**；建议对齐 RAF-DB 7 类
   （neutral/happy/sad/anger/surprise/fear/disgust）。最终类别可配置。
4. **模型（学生）**：**MobileNetV3-Small** 或 **EfficientNet-B0/-lite0**，int8。可直接采用/微调 **HSEmotion(EmotiEffLib)**
   的 AffectNet 预训练 EfficientNet-B0（移动端验证充分、自带 ONNX）。
5. **知识蒸馏**：强老师（POSTER++/DDAMFN 或 HSEmotion B0）→ 小学生。**复用现有 `optimize_distill.py` 蒸馏代码**。
6. **量化**：先 **INT8 PTQ**（per-channel 权重 + 代表性标定集，标定集取真实摄像头帧分布）；小模型 PTQ 掉点明显时
   升级 **QAT** 恢复精度（调研结论：小/敏感模型 PTQ 易掉点，QAT 更稳）。
7. **导出链路**：PyTorch → **ONNX(opset 13)** →（RKNN-Toolkit2 / ncnn / onnxruntime）。
   注意 RKNN 编译器偏好 opset 11–13、需固定输入尺寸、Resize 行为敏感。
8. **时序平滑**：设备端对逐帧 logits 做滑动平均/EMA + 置信度阈值 + 滞回，避免表情"抖动"，提升体感稳定性。

## 4. 架构（三层，清晰解耦）

```
┌─ 离线训练管线（开发机/GPU）──────────────────────────────┐
│ 数据准备(YuNet裁脸/对齐) → 训练老师 → 蒸馏学生            │
│ → 导出 ONNX(opset13) → INT8 量化(PTQ/QAT)+标定           │
│ → 产物: emotion_int8.onnx (+ 标签/前处理元信息)          │
└──────────────────────────────────────────────────────────┘
                         │ ONNX 产物
                         ▼
┌─ 硬件适配层（可插拔后端）────────────────────────────────┐
│ EmotionEngine 接口： onnxruntime(PC/通用) | rknn(瑞芯微) │
│                      | ncnn(纯ARM CPU)                   │
└──────────────────────────────────────────────────────────┘
                         │ 统一接口
                         ▼
┌─ 设备端运行时（机器人）──────────────────────────────────┐
│ 摄像头取帧 → YuNet 检测裁脸 → EmotionEngine 推理         │
│ → 时序平滑 → 情绪事件(label+score) → 驱动屏幕/语音反应   │
└──────────────────────────────────────────────────────────┘
```

## 5. 模块划分（接口边界）

- `src/face/detector.py`：`FaceDetector`（YuNet 封装）→ 输入帧，输出对齐裁剪的人脸 ROI。
- `src/data/prepare.py`：批量裁脸/对齐 + 划分，产出训练用人脸库。
- `src/model.py`（扩展现有注册表）：加入 mobilenetv3_small / effnet-lite，登记输入尺寸。
- `src/distill`（复用 `optimize_distill.py`）：老师→学生蒸馏。
- `src/export/to_onnx.py`：固定尺寸导出 ONNX(opset13) + 校验数值一致。
- `src/export/quantize.py`：INT8 PTQ（onnxruntime quant / 标定集），可选 QAT 入口。
- `src/engine/`：`EmotionEngine` 抽象 + `onnxruntime/rknn/ncnn` 三后端实现（统一 `infer(face)->probs`）。
- `src/runtime/loop.py`：摄像头循环 + 平滑 + 事件输出（PC Demo 用 OpenCV 窗口可视化）。
- Flask `app.py`：退役为**开发期可视化/调试器**（非出厂路径），或最终移除。

## 6. 现有代码：复用 / 退役

- **复用**：模型工厂注册表、`inference.py` 自描述 checkpoint、蒸馏/量化/剪枝、transforms、config、logging、测试基建。
- **改造**：`config` 增加类别集合（含 neutral）、人脸输入尺寸（如 112×112）、后端选择。
- **退役/降级**：Flask 在线服务从"产品"降为"调试器"；FER-2013 相关默认值移除。

## 7. 分期（YAGNI：先交付硬件无关核心）

- **Phase 1（硬件无关核心，本机可开发验证）**：YuNet 裁脸 → 训练/蒸馏脚本就绪 → ONNX 导出 + INT8 PTQ 标定
  → `onnxruntime` 后端 → **PC 摄像头 Demo 跑通完整闭环**（取帧→裁脸→推理→平滑→情绪事件）。
  产出一个可在普通电脑上演示的"情绪核心"，证明链路。
- **Phase 2（目标板适配，待硬件锁定后）**：RKNN（或 ncnn）后端 + 板上实测 + QAT 精度恢复 + 功耗/FPS 调优
  + 与机器人屏幕/语音的事件对接协议。

## 8. 测试策略

- 单测（本机，无需数据/GPU）：YuNet 在合成/样例图上的检测、裁剪尺寸；ONNX 导出与 PyTorch 数值一致性（allclose）；
  PTQ 量化前后输出偏差阈值；EmotionEngine 各后端同一输入一致；时序平滑逻辑；类别映射。
- 集成：合成帧驱动 runtime loop，断言产出合法情绪事件。
- 设备端（Phase 2）：板上 FPS/延迟/功耗实测、PTQ vs QAT 精度对比。

## 9. 风险与对策

- **小模型 PTQ 掉点** → 预留 QAT 路径；per-channel + 真实分布标定集。
- **RKNN 算子/opset 兼容坑** → 锁 opset 13、固定输入尺寸、导出后用 RKNN-Toolkit2 模拟器校验。
- **数据集许可** → RAF-DB/AffectNet 多为学术许可，量产前需确认商用授权；必要时改用可商用集或自采。
- **AffectNet 8 类精度天然偏低** → 陪伴场景可合并细粒度类别（如 fear/surprise）或减类，换取稳定体感。
- **硬件未锁** → 以 ONNX 为中心、后端可插拔，推迟硬件耦合到 Phase 2。

## 来源（调研）

- HSEmotion / EmotiEffLib（移动端 FER，PyTorch+ONNX）：https://github.com/av-savchenko/hsemotion-onnx ，https://github.com/sb-ai-lab/EmotiEffLib
- YuNet 轻量人脸检测：https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf
- 轻量 FER 综述/基准（RAF-DB/AffectNet 数值）：https://link.springer.com/article/10.1007/s10791-025-09699-8 ，https://arxiv.org/pdf/2311.02910
- RKNN-Toolkit2 / 瑞芯微 NPU 部署：https://docs.ultralytics.com/integrations/rockchip-rknn ，https://medium.com/@zediot/deploying-yolov8-on-rk3566-a-deep-dive-into-model-conversion-quantization-and-real-world-383d3de7e39a
- PyTorch→ONNX INT8 PTQ/QAT 边缘最佳实践：https://inference.net/content/post-training-quantization ，https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html
