import os
from dataclasses import replace
from unittest.mock import patch

import pytest

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


def test_ensure_failed_download_leaves_no_partial_file(tmp_path):
    dest = tmp_path / 'sub' / 'broken.onnx'
    asset = replace(assets.YUNET, dest=str(dest))

    def fail_midway(url, d):
        open(d, 'wb').write(b'half')  # 模拟写了一半后中断
        raise OSError('网络中断')

    with patch('urllib.request.urlretrieve', side_effect=fail_midway), \
            pytest.raises(OSError):
        assets.ensure(asset)
    # 目标与临时文件都不应存在，下次调用会重新下载
    assert not os.path.exists(str(dest))
    assert not os.path.exists(str(dest) + '.part')
