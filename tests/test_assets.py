import os
from dataclasses import replace
from unittest.mock import patch

from src import assets


def test_registry_metadata_is_sane():
    for a in assets.ALL_ASSETS:
        assert a.url.startswith('https://')
        assert a.dest.endswith('.onnx')
    em = assets.DEFAULT_EMOTION
    assert len(em.labels) == 8
    assert em.labels[em.labels.index('Neutral')] == 'Neutral'  # 陪伴场景必备 neutral
    assert em.input_size == 224


def test_ensure_skips_download_when_file_exists(tmp_path):
    dest = tmp_path / 'cached.onnx'
    dest.write_bytes(b'x')
    asset = replace(assets.YUNET, dest=str(dest))
    with patch('urllib.request.urlretrieve') as fake:
        assert assets.ensure(asset) == str(dest)
        fake.assert_not_called()


def test_ensure_downloads_when_missing(tmp_path):
    dest = tmp_path / 'sub' / 'new.onnx'
    asset = replace(assets.YUNET, dest=str(dest))
    with patch('urllib.request.urlretrieve') as fake:
        fake.side_effect = lambda url, d: open(d, 'wb').write(b'x')
        assert assets.ensure(asset) == str(dest)
        fake.assert_called_once()
        assert os.path.exists(str(dest))
