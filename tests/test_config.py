from pathlib import Path
import src.config as config


def test_classes_and_index():
    assert config.CLASSES == ['anger', 'fear', 'happy', 'sad', 'surprise']
    assert config.NUM_CLASSES == 5
    assert config.CLASS_TO_IDX['happy'] == 2


def test_zh_and_emoji_cover_all_classes():
    for cls in config.CLASSES:
        assert cls in config.CLASS_NAMES_ZH
        assert cls in config.CLASS_EMOJIS


def test_paths_are_under_project_root():
    root = config.PROJECT_ROOT
    assert isinstance(root, Path)
    assert config.RAW_DIR == root / 'data' / 'raw'
    assert config.TRAIN_DIR == root / 'data' / 'train'
    assert config.VAL_DIR == root / 'data' / 'val'
    assert config.MODELS_DIR == root / 'models'
    assert config.RESULTS_DIR == root / 'results'


def test_train_config_defaults():
    cfg = config.TrainConfig()
    assert cfg.model_type == 'resnet18'
    assert cfg.num_epochs == 20
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.001
    assert cfg.device == 'auto'
