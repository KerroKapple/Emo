import logging
from src.logging_setup import get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger('emotion.test')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'emotion.test'


def test_get_logger_has_single_handler():
    logger = get_logger('emotion.handlers')
    # 多次获取不应重复挂 handler
    logger2 = get_logger('emotion.handlers')
    assert len(logger2.handlers) == 1
    assert logger is logger2


def test_logger_level_is_info():
    logger = get_logger('emotion.level')
    assert logger.level == logging.INFO
