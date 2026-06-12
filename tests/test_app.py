import io
import base64

import pytest
from PIL import Image

import app as app_module
import src.config as config
from src.model import get_model
from src.inference import save_checkpoint


@pytest.fixture
def client():
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_service(monkeypatch):
    """每个测试用全新（未加载）的 ModelService，避免全局状态污染"""
    monkeypatch.setattr(app_module, 'service', app_module.ModelService())


def _b64_image(color=(120, 30, 200)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def test_status_reports_not_loaded(client):
    data = client.get('/api/status').get_json()
    assert data['model_loaded'] is False
    assert data['classes'] == config.CLASSES


def test_predict_without_model_returns_400(client):
    r = client.post('/api/predict', json={'image': _b64_image()})
    assert r.status_code == 400


def test_predict_after_load_returns_distribution(client, tmp_path):
    model = get_model('cnn', num_classes=config.NUM_CLASSES, pretrained=False)
    path = str(tmp_path / 'best_model_cnn.pth')
    save_checkpoint(model, 'cnn', path, val_acc=80.0)
    app_module.service.load(path)

    r = client.post('/api/predict', json={'image': _b64_image()})
    assert r.status_code == 200
    data = r.get_json()
    assert data['predicted_class'] in config.CLASSES
    total = sum(p['probability'] for p in data['probabilities'].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_missing_image_returns_400(client, tmp_path):
    model = get_model('cnn', num_classes=config.NUM_CLASSES, pretrained=False)
    path = str(tmp_path / 'best_model_cnn.pth')
    save_checkpoint(model, 'cnn', path, val_acc=80.0)
    app_module.service.load(path)

    r = client.post('/api/predict', json={})
    assert r.status_code == 400


def test_load_model_missing_filename_returns_400(client):
    r = client.post('/api/load_model', json={})
    assert r.status_code == 400


@pytest.mark.parametrize('bad', ['../secret.pth', '..\\secret.pth', 'C:/x/y.pth', 'model.bin'])
def test_load_model_rejects_traversal_and_bad_suffix(client, bad):
    r = client.post('/api/load_model', json={'model_filename': bad})
    assert r.status_code == 400


def test_load_model_nonexistent_returns_404(client):
    r = client.post('/api/load_model', json={'model_filename': 'no_such_model.pth'})
    assert r.status_code == 404


def test_predict_invalid_base64_returns_400(client, tmp_path):
    model = get_model('cnn', num_classes=config.NUM_CLASSES, pretrained=False)
    path = str(tmp_path / 'best_model_cnn.pth')
    save_checkpoint(model, 'cnn', path, val_acc=80.0)
    app_module.service.load(path)

    r = client.post('/api/predict', json={'image': 'data:image/png;base64,not-an-image'})
    assert r.status_code == 400
