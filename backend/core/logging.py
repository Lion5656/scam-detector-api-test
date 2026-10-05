import logging


def configure_logging():
    """配置日誌系統"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True
)