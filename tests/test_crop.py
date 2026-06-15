import numpy as np
from src.face.crop import crop_face


def _img(h=200, w=200):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_crop_returns_requested_size():
    out = crop_face(_img(), (50, 50, 100, 100), size=112)
    assert out.shape == (112, 112, 3)


def test_crop_clamps_at_image_border():
    out = crop_face(_img(200, 200), (180, 180, 100, 100), size=64, margin=0.3)
    assert out.shape == (64, 64, 3)


def test_crop_empty_box_returns_none():
    assert crop_face(_img(), (10, 10, 0, 0)) is None
