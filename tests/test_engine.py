import numpy as np
from tests._fixtures import export_tiny_onnx
from src.engine.onnx_engine import OnnxRuntimeEngine

LABELS = ['anger', 'fear', 'happy', 'sad', 'surprise']


def test_engine_infers_probability_vector(tmp_path):
    path = export_tiny_onnx(str(tmp_path / 'tiny.onnx'), num_classes=len(LABELS))
    engine = OnnxRuntimeEngine(path, LABELS, input_size=112)
    face = (np.random.rand(80, 70, 3) * 255).astype(np.uint8)  # 任意尺寸 BGR
    probs = engine.infer(face)
    assert probs.shape == (len(LABELS),)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert engine.labels == LABELS


def test_preprocess_outputs_nchw(tmp_path):
    path = export_tiny_onnx(str(tmp_path / 'tiny.onnx'), num_classes=len(LABELS))
    engine = OnnxRuntimeEngine(path, LABELS, input_size=112)
    batch = engine.preprocess((np.random.rand(50, 50, 3) * 255).astype(np.uint8))
    assert batch.shape == (1, 3, 112, 112)
    assert batch.dtype == np.float32
