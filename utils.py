from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from functools import wraps
import logging
from datetime import datetime
import time
import os
import sys
import shutil
import requests
import socket
from typing import Callable, Any, Type, Union, List

# プロジェクトルートディレクトリを追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config

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

# 再試行対象の例外クラス
NETWORK_EXCEPTIONS = (
    requests.exceptions.RequestException,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.HTTPError,
    socket.timeout,
    socket.gaierror,
    ConnectionError,
    TimeoutError
)

API_EXCEPTIONS = (
    requests.exceptions.HTTPError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)

def retry_on_error(
    max_attempts: int = None,
    wait_min: int = None,
    wait_max: int = None,
    retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = None
) -> Callable:
    """
    拡張リトライデコレータ
    
    Args:
        max_attempts: 最大試行回数（Noneの場合はconfig値を使用）
        wait_min: 最小待機時間（秒）
        wait_max: 最大待機時間（秒）
        retry_exceptions: 再試行する例外のクラス
    """
    config = Config()
    
    if max_attempts is None:
        max_attempts = config.RETRY_ATTEMPTS
    if wait_min is None:
        wait_min = config.RETRY_WAIT_MIN
    if wait_max is None:
        wait_max = config.RETRY_WAIT_MAX
    if retry_exceptions is None:
        retry_exceptions = NETWORK_EXCEPTIONS
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
            retry=retry_if_exception_type(retry_exceptions)
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                logger.debug(f"Attempting to execute {func.__name__}")
                result = func(*args, **kwargs)
                logger.debug(f"Successfully executed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Exception type: {type(e).__name__}")
                raise
        return wrapper
    return decorator

def retry_api_call(max_attempts: int = None) -> Callable:
    """API呼び出し専用のリトライデコレータ"""
    return retry_on_error(
        max_attempts=max_attempts,
        retry_exceptions=API_EXCEPTIONS
    )

def retry_network_operation(max_attempts: int = None) -> Callable:
    """ネットワーク操作専用のリトライデコレータ"""
    return retry_on_error(
        max_attempts=max_attempts,
        retry_exceptions=NETWORK_EXCEPTIONS
    )

def retry_file_operation(max_attempts: int = 3) -> Callable:
    """ファイル操作専用のリトライデコレータ"""
    return retry_on_error(
        max_attempts=max_attempts,
        retry_exceptions=(IOError, OSError, PermissionError)
    )

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

def validate_dataframe(df, required_columns: List[str] = None, min_rows: int = 1) -> bool:
    """
    DataFrameの検証
    
    Args:
        df: 検証対象のDataFrame
        required_columns: 必須カラムのリスト
        min_rows: 最小行数
    
    Returns:
        bool: 検証結果
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    logger.debug(f"DataFrame validation passed: {len(df)} rows, columns: {list(df.columns)}")
    return True

def validate_market_data(data: dict) -> bool:
    """
    マーケットデータの検証
    
    Args:
        data: マーケットデータ辞書
    
    Returns:
        bool: 検証結果
    """
    if not data:
        logger.error("Market data is empty")
        return False
    
    required_fields = ['current', 'change', 'change_percent']
    for ticker, ticker_data in data.items():
        if not isinstance(ticker_data, dict):
            logger.error(f"Invalid data type for ticker {ticker}: {type(ticker_data)}")
            return False
        
        missing_fields = [field for field in required_fields if field not in ticker_data]
        if missing_fields:
            logger.error(f"Missing fields for ticker {ticker}: {missing_fields}")
            return False
    
    logger.debug(f"Market data validation passed for {len(data)} tickers")
    return True

def validate_news_data(news_list: List[dict]) -> bool:
    """
    ニュースデータの検証
    
    Args:
        news_list: ニュース記事のリスト
    
    Returns:
        bool: 検証結果
    """
    if not news_list:
        logger.warning("News list is empty")
        return True  # 空でも有効とする
    
    required_fields = ['title', 'url', 'published_jst', 'country']
    for i, article in enumerate(news_list):
        if not isinstance(article, dict):
            logger.error(f"Invalid article type at index {i}: {type(article)}")
            return False
        
        missing_fields = [field for field in required_fields if field not in article]
        if missing_fields:
            logger.error(f"Missing fields in article {i}: {missing_fields}")
            return False
        
        # URLの基本検証
        if not isinstance(article['url'], str) or not article['url'].startswith('http'):
            logger.error(f"Invalid URL in article {i}: {article['url']}")
            return False
    
    logger.debug(f"News data validation passed for {len(news_list)} articles")
    return True

def timeout_wrapper(timeout_seconds: int = None):
    """
    タイムアウト機能付きデコレータ
    
    Args:
        timeout_seconds: タイムアウト時間（秒）
    """
    config = Config()
    if timeout_seconds is None:
        timeout_seconds = config.TIMEOUT_SECONDS
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # タイムアウト設定
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # タイムアウトをクリア
                return result
            except Exception as e:
                signal.alarm(0)  # タイムアウトをクリア
                logger.error(f"Function {func.__name__} failed: {str(e)}")
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

def safe_request(url: str, timeout: int = None, max_retries: int = 3, **kwargs) -> requests.Response:
    """
    安全なHTTPリクエスト実行
    
    Args:
        url: リクエストURL
        timeout: タイムアウト時間
        max_retries: 最大リトライ数
        **kwargs: requestsライブラリへの追加パラメータ
    
    Returns:
        requests.Response: レスポンスオブジェクト
    """
    config = Config()
    if timeout is None:
        timeout = config.HTTP_REQUEST_TIMEOUT
    
    @retry_network_operation(max_attempts=max_retries)
    def _make_request():
        response = requests.get(url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    
    return _make_request()

def log_performance_metrics(func_name: str, execution_time: float, **metrics):
    """
    パフォーマンスメトリクスのログ出力
    
    Args:
        func_name: 関数名
        execution_time: 実行時間
        **metrics: 追加メトリクス
    """
    metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'function': func_name,
        'execution_time_seconds': round(execution_time, 3),
        **metrics
    }
    
    logger.info(f"METRICS: {metrics_data}")

class DataValidator:
    """データ検証クラス"""
    
    @staticmethod
    def validate_ticker_data(ticker: str, data: dict) -> bool:
        """個別ティッカーデータの検証"""
        if not data:
            logger.error(f"No data for ticker {ticker}")
            return False
        
        # 数値データの検証
        numeric_fields = ['current', 'change', 'change_percent']
        for field in numeric_fields:
            if field in data and data[field] != 'N/A':
                try:
                    float(str(data[field]).replace('%', '').replace(',', ''))
                except (ValueError, TypeError):
                    logger.error(f"Invalid numeric value for {ticker}.{field}: {data[field]}")
                    return False
        
        return True
    
    @staticmethod
    def validate_chart_data(data, ticker_name: str) -> bool:
        """チャートデータの検証"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return validate_dataframe(data, required_columns, min_rows=1)
