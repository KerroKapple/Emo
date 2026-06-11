"""app.py - Flask 表情识别服务"""

import io
import os
import base64
import threading

import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

import src.config as config
from src.logging_setup import get_logger
from src.transforms import build_transform
from src.inference import load_model_from_checkpoint, predict_probs

logger = get_logger('emotion.app')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES
CORS(app)


class ModelService:
    """线程安全地持有当前模型及其预处理管线"""

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.info = None
        self.transform = None
        self.device = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self, model_path):
        model, info = load_model_from_checkpoint(model_path, device='auto')
        transform = build_transform(info['model_type'], train=False)
        with self._lock:
            self.model, self.info, self.transform, self.device = model, info, transform, info['device']
        logger.info("已加载模型 %s | 设备 %s", info['model_type'], info['device'])
        return info

    def predict(self, pil_image):
        with self._lock:
            if self.model is None:
                raise RuntimeError('模型未加载')
            return predict_probs(self.model, pil_image, self.transform, self.device)


service = ModelService()


def _read_checkpoint_meta(path):
    """读取 checkpoint 元信息用于列表展示"""
    ckpt = torch.load(path, map_location='cpu')
    return {
        'model_type': ckpt.get('model_type', 'unknown'),
        'accuracy': ckpt.get('accuracy'),
        'is_quantized': ckpt.get('quantized', False),
        'is_pruned': ckpt.get('pruned', False),
        'is_distilled': ckpt.get('distilled', False),
    }


def _build_response(probs):
    classes = config.CLASSES
    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    pred = classes[pred_idx]
    return {
        'predicted_class': pred,
        'predicted_class_zh': config.CLASS_NAMES_ZH[pred],
        'emoji': config.CLASS_EMOJIS[pred],
        'confidence': float(probs[pred_idx]),
        'probabilities': {
            cls: {
                'en': cls,
                'zh': config.CLASS_NAMES_ZH[cls],
                'emoji': config.CLASS_EMOJIS[cls],
                'probability': float(probs[i]),
            }
            for i, cls in enumerate(classes)
        },
    }


def _read_request_image():
    """从 multipart 文件或 base64 字段解析出 PIL 图像"""
    if 'file' in request.files:
        return Image.open(request.files['file'].stream).convert('RGB')

    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image')
    if image_data:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_available_models():
    models_dir = str(config.MODELS_DIR)
    if not os.path.exists(models_dir):
        return jsonify({'error': '模型文件夹不存在'}), 404

    models = []
    for f in sorted(os.listdir(models_dir)):
        if not f.endswith('.pth'):
            continue
        path = os.path.join(models_dir, f)
        try:
            meta = _read_checkpoint_meta(path)
        except Exception as e:
            logger.warning("读取 checkpoint 失败 %s: %s", f, e)
            continue
        models.append({
            'filename': f,
            'size_mb': round(os.path.getsize(path) / (1024 * 1024), 2),
            **meta,
        })
    return jsonify({'models': models})


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    data = request.get_json(silent=True) or {}
    filename = data.get('model_filename')
    if not filename:
        return jsonify({'error': '缺少 model_filename'}), 400

    model_path = os.path.join(str(config.MODELS_DIR), filename)
    if not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 404

    try:
        info = service.load(model_path)
        return jsonify({
            'success': True,
            'model_type': info['model_type'],
            'device': str(info['device']),
            'accuracy': info['accuracy'],
        })
    except Exception as e:
        logger.exception("加载模型失败")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    if not service.loaded:
        return jsonify({'error': '请先加载模型'}), 400

    try:
        image = _read_request_image()
    except Exception as e:
        logger.warning("图片解析失败: %s", e)
        return jsonify({'error': '图片解析失败'}), 400

    if image is None:
        return jsonify({'error': '未提供图片'}), 400

    try:
        probs = service.predict(image)
        return jsonify(_build_response(probs))
    except Exception as e:
        logger.exception("预测失败")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': service.loaded,
        'model_type': service.info['model_type'] if service.loaded else None,
        'device': str(service.device) if service.device else None,
        'classes': config.CLASSES,
        'classes_zh': config.CLASS_NAMES_ZH,
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({'error': f'上传文件超过 {config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制'}), 413


def _load_default_model():
    """启动时尝试加载首个 best_model_*.pth"""
    models_dir = config.MODELS_DIR
    if not models_dir.exists():
        return
    for f in sorted(models_dir.glob('best_model_*.pth')):
        try:
            service.load(str(f))
            logger.info("默认模型已加载: %s", f.name)
        except Exception as e:
            logger.warning("默认模型加载失败: %s", e)
        break


if __name__ == '__main__':
    _load_default_model()

    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    logger.info("启动服务 http://%s:%d (debug=%s)", host, port, debug)

    if debug:
        app.run(debug=True, host=host, port=port)
    else:
        from waitress import serve
        serve(app, host=host, port=port)
