"""onnx_engine.py - onnxruntime CPU 后端"""

import onnxruntime as ort

from src.engine.base import EmotionEngine


class OnnxRuntimeEngine(EmotionEngine):
    def __init__(self, model_path, labels, providers=('CPUExecutionProvider',), **kw):
        super().__init__(labels, **kw)
        self._sess = ort.InferenceSession(model_path, providers=list(providers))
        self._input = self._sess.get_inputs()[0].name

    def _forward(self, batch):
        return self._sess.run(None, {self._input: batch})[0]
