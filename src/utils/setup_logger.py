"""
簡易ロガー設定ユーティリティ
logger.pyでエラーが発生する場合のフォールバック
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str, level: str = "INFO"):
    """簡易ロガーの設定"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    # ログレベル設定
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（簡易版）
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / "market_report.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to setup file handler: {e}")
    
    return logger