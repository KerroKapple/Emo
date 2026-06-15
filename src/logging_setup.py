"""logging_setup.py - 统一日志配置"""

import logging

_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_DATEFMT = '%H:%M:%S'


def get_logger(name: str) -> logging.Logger:
    """返回配置好的命名 logger；重复调用同名不会重复挂 handler"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
