from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps
import logging
from datetime import datetime
import time
import os
import sys

# プロジェクトルートディレクトリを追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from typing import Callable, Any

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ファイルハンドラーの設定
file_handler = logging.FileHandler('execution.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ストリームハンドラーの設定
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def retry_on_error(
    max_attempts: int = 3,
    wait_min: int = 4,
    wait_max: int = 10
) -> Callable:
    """リトライデコレータ"""
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max)
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def measure_time(func: Callable) -> Callable:
    """実行時間計測デコレータ"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def check_disk_space(min_space_mb: int = 1024) -> bool:
    """ディスク空き容量をチェック"""
    total, used, free = shutil.disk_usage("/")
    free_mb = free // (1024 * 1024)
    if free_mb < min_space_mb:
        logger.warning(f"Low disk space: {free_mb}MB available")
        return False
    return True

def check_memory_usage(max_memory_mb: int = 512) -> bool:
    """メモリ使用量をチェック"""
    import psutil
    memory = psutil.virtual_memory()
    if memory.percent > 90:  # 90%以上使用している場合
        logger.warning(f"High memory usage: {memory.percent}%")
        return False
    return True
