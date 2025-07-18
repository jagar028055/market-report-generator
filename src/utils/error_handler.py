"""
エラーハンドリングとリトライ機能を提供するモジュール
"""

import logging
import traceback
from functools import wraps
from typing import Callable, Any, Type, Union, List, Optional
from datetime import datetime
import time
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

from .exceptions import (
    MarketReportException,
    NetworkError,
    TimeoutError,
    APIError,
    RateLimitError,
    DataFetchError,
    ValidationError,
    ServiceUnavailableError
)


class ErrorHandler:
    """エラーハンドリングを統一するクラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history = []
    
    def handle_error(self, error: Exception, context: dict = None) -> None:
        """エラーを処理し、適切にログに記録する"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {}
        }
        
        # エラー履歴に追加
        self.error_history.append(error_info)
        
        # ログレベルを決定
        if isinstance(error, (NetworkError, TimeoutError, APIError)):
            self.logger.warning(f"Network/API error: {error}")
        elif isinstance(error, ValidationError):
            self.logger.error(f"Validation error: {error}")
        elif isinstance(error, MarketReportException):
            self.logger.error(f"Market report error: {error}")
        else:
            self.logger.error(f"Unexpected error: {error}")
        
        # デバッグモードの場合は詳細なトレースバックを記録
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Error traceback: {traceback.format_exc()}")
    
    def get_error_summary(self) -> dict:
        """エラーの概要を取得"""
        if not self.error_history:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = {}
        for error_info in self.error_history:
            error_type = error_info['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-5:]  # 最新5件
        }
    
    def clear_history(self) -> None:
        """エラー履歴をクリア"""
        self.error_history.clear()


class RetryHandler:
    """リトライ機能を提供するクラス"""
    
    NETWORK_EXCEPTIONS = (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
        NetworkError,
        TimeoutError,
    )
    
    API_EXCEPTIONS = (
        requests.exceptions.HTTPError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        APIError,
    )
    
    def __init__(self, error_handler: ErrorHandler = None):
        self.error_handler = error_handler or ErrorHandler()
    
    def create_retry_decorator(
        self,
        max_attempts: int = 3,
        wait_min: int = 1,
        wait_max: int = 60,
        retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = None
    ) -> Callable:
        """リトライデコレータを作成"""
        
        if retry_exceptions is None:
            retry_exceptions = self.NETWORK_EXCEPTIONS
        
        def decorator(func: Callable) -> Callable:
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
                retry=retry_if_exception_type(retry_exceptions)
            )
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.error_handler.handle_error(e, {
                        'function': func.__name__,
                        'args': str(args)[:200],  # 長すぎる場合は切り詰め
                        'kwargs': str(kwargs)[:200]
                    })
                    raise
            return wrapper
        return decorator
    
    def retry_network_operation(self, max_attempts: int = 3) -> Callable:
        """ネットワーク操作専用のリトライデコレータ"""
        return self.create_retry_decorator(
            max_attempts=max_attempts,
            retry_exceptions=self.NETWORK_EXCEPTIONS
        )
    
    def retry_api_call(self, max_attempts: int = 3) -> Callable:
        """API呼び出し専用のリトライデコレータ"""
        return self.create_retry_decorator(
            max_attempts=max_attempts,
            retry_exceptions=self.API_EXCEPTIONS
        )
    
    def retry_data_fetch(self, max_attempts: int = 3) -> Callable:
        """データ取得専用のリトライデコレータ"""
        return self.create_retry_decorator(
            max_attempts=max_attempts,
            retry_exceptions=(DataFetchError,) + self.NETWORK_EXCEPTIONS
        )


def handle_api_response(response: requests.Response) -> dict:
    """API レスポンスを処理し、エラーがあれば適切な例外を発生させる"""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            raise RateLimitError(
                f"Rate limit exceeded: {response.status_code}",
                retry_after=int(retry_after) if retry_after else None
            )
        elif response.status_code >= 500:
            raise APIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code
            )
        elif response.status_code >= 400:
            raise APIError(
                f"Client error: {response.status_code}",
                status_code=response.status_code
            )
        else:
            raise APIError(f"HTTP error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Network error: {str(e)}")
    except json.JSONDecodeError:
        raise APIError("Invalid JSON response")


def with_error_handling(
    logger: Optional[logging.Logger] = None,
    reraise: bool = True
) -> Callable:
    """エラーハンドリング機能を追加するデコレータ"""
    
    error_handler = ErrorHandler(logger)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, {
                    'function': func.__name__,
                    'module': func.__module__
                })
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return=None,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Any:
    """関数を安全に実行し、エラーが発生した場合はデフォルト値を返す"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return


class CircuitBreaker:
    """サーキットブレーカーパターンの実装"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise ServiceUnavailableError(
                        "Service temporarily unavailable (circuit breaker open)"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _on_success(self):
        """成功時の処理"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """失敗時の処理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


# 下位互換性のための既存関数
def handle_step_error(step_name: str, error_class: type = MarketReportException):
    """
    処理ステップのエラーハンドリングデコレータ（下位互換性のため）
    
    Args:
        step_name: 処理ステップ名
        error_class: 発生させるエラークラス
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = f"{step_name}でエラーが発生しました: {str(e)}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                raise error_class(error_message) from e
        return wrapper
    return decorator


def log_and_reraise(error: Exception, step_name: str, logger: Optional[logging.Logger] = None):
    """
    エラーをログに記録して再発生させる（下位互換性のため）
    
    Args:
        error: 発生したエラー
        step_name: 処理ステップ名
        logger: ロガー（指定されない場合はデフォルトロガーを使用）
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_message = f"{step_name}でエラーが発生しました: {str(error)}"
    logger.error(error_message)
    logger.error(traceback.format_exc())
    raise


def create_error_summary(errors: list) -> str:
    """
    エラーリストからサマリーを作成（下位互換性のため）
    
    Args:
        errors: エラーのリスト
        
    Returns:
        エラーサマリー文字列
    """
    if not errors:
        return "エラーはありません"
    
    summary = "以下のエラーが発生しました:\n"
    for i, error in enumerate(errors, 1):
        summary += f"{i}. {str(error)}\n"
    
    return summary


# グローバルなエラーハンドラーインスタンス
global_error_handler = ErrorHandler()
global_retry_handler = RetryHandler(global_error_handler)