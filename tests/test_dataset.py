import inspect
import os
import tempfile

from src.dataset import EmotionDataset
import src.config as config


def test_dataset_uses_config_classes():
    # 通过 __init__ 默认构造一个空目录数据集，验证类别来自 config
    with tempfile.TemporaryDirectory() as d:
        for cls in config.CLASSES:
            os.makedirs(os.path.join(d, cls))
        ds = EmotionDataset(d, clean_on_load=False)
        assert ds.classes == config.CLASSES
        assert ds.class_to_idx == config.CLASS_TO_IDX


def test_clean_on_load_default_is_false():
    sig = inspect.signature(EmotionDataset.__init__)
    assert sig.parameters['clean_on_load'].default is False
