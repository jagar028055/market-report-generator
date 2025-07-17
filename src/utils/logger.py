"""
構造化ロギングとメトリクス収集モジュール
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config

class JSONFormatter(logging.Formatter):
    """JSON形式のログフォーマッター"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread if hasattr(record, 'thread') else None,
        }
        
        # 例外情報があれば追加
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # カスタム属性を追加
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class MetricsLogger:
    """メトリクス収集とログ出力クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('market_report_metrics')
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        logger.handlers.clear()  # 既存のハンドラーをクリア
        
        # JSON形式のフォーマッター
        json_formatter = JSONFormatter()
        
        # ファイルハンドラー（ローテーション対応）
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.LOG_FILE,
            maxBytes=self.config.LOG_MAX_BYTES,
            backupCount=self.config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
        
        # コンソールハンドラー（開発環境のみ）
        if self.config.ENVIRONMENT == "development":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(json_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_function_metrics(self, func_name: str, execution_time: float, **kwargs):
        """関数実行メトリクスのログ出力"""
        self.logger.info(
            "Function execution metrics",
            extra={
                'metric_type': 'function_execution',
                'function_name': func_name,
                'execution_time_seconds': round(execution_time, 3),
                **kwargs
            }
        )
    
    def log_api_metrics(self, api_name: str, endpoint: str, status_code: int, 
                       response_time: float, **kwargs):
        """API呼び出しメトリクスのログ出力"""
        self.logger.info(
            "API call metrics",
            extra={
                'metric_type': 'api_call',
                'api_name': api_name,
                'endpoint': endpoint,
                'status_code': status_code,
                'response_time_seconds': round(response_time, 3),
                'success': status_code < 400,
                **kwargs
            }
        )
    
    def log_data_metrics(self, data_type: str, record_count: int, 
                        data_size_bytes: Optional[int] = None, **kwargs):
        """データ処理メトリクスのログ出力"""
        metrics = {
            'metric_type': 'data_processing',
            'data_type': data_type,
            'record_count': record_count,
            **kwargs
        }
        
        if data_size_bytes is not None:
            metrics['data_size_bytes'] = data_size_bytes
            metrics['data_size_mb'] = round(data_size_bytes / (1024 * 1024), 2)
        
        self.logger.info(
            "Data processing metrics",
            extra=metrics
        )
    
    def log_error_metrics(self, error_type: str, error_message: str, 
                         function_name: str, **kwargs):
        """エラーメトリクスのログ出力"""
        self.logger.error(
            "Error occurred",
            extra={
                'metric_type': 'error',
                'error_type': error_type,
                'error_message': error_message,
                'function_name': function_name,
                **kwargs
            }
        )
    
    def log_system_metrics(self):
        """システムメトリクスのログ出力"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            
            # ディスク使用率
            disk = psutil.disk_usage('/')
            
            self.logger.info(
                "System metrics",
                extra={
                    'metric_type': 'system',
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': round(memory.available / (1024 * 1024)),
                    'disk_free_gb': round(disk.free / (1024 * 1024 * 1024), 2),
                    'disk_percent': round((disk.used / disk.total) * 100, 1)
                }
            )
        except ImportError:
            self.logger.warning("psutil not available for system metrics")
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def start_timer(self, timer_name: str):
        """タイマー開始"""
        self.metrics[timer_name] = time.time()
    
    def end_timer(self, timer_name: str) -> float:
        """タイマー終了して実行時間を返す"""
        if timer_name not in self.metrics:
            raise ValueError(f"Timer '{timer_name}' not started")
        
        execution_time = time.time() - self.metrics[timer_name]
        del self.metrics[timer_name]
        return execution_time

# グローバルなメトリクスロガーインスタンス
_metrics_logger = None

def get_metrics_logger() -> MetricsLogger:
    """メトリクスロガーのシングルトンインスタンスを取得"""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger

def log_execution_time(func_name: Optional[str] = None):
    """実行時間をログに記録するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            metrics_logger = get_metrics_logger()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                metrics_logger.log_function_metrics(
                    func_name=name,
                    execution_time=execution_time,
                    success=True
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                metrics_logger.log_function_metrics(
                    func_name=name,
                    execution_time=execution_time,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                metrics_logger.log_error_metrics(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    function_name=name
                )
                raise
        return wrapper
    return decorator

def log_api_call(api_name: str):
    """API呼び出しをログに記録するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics_logger = get_metrics_logger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # レスポンスから情報を取得（可能な場合）
                status_code = getattr(result, 'status_code', 200)
                endpoint = kwargs.get('url', 'unknown')
                
                metrics_logger.log_api_metrics(
                    api_name=api_name,
                    endpoint=endpoint,
                    status_code=status_code,
                    response_time=response_time
                )
                return result
            except Exception as e:
                response_time = time.time() - start_time
                metrics_logger.log_api_metrics(
                    api_name=api_name,
                    endpoint=kwargs.get('url', 'unknown'),
                    status_code=getattr(e, 'status_code', 500),
                    response_time=response_time,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics_logger = get_metrics_logger()
        self.start_time = None
        self.checkpoints = {}
    
    def start_monitoring(self):
        """監視開始"""
        self.start_time = time.time()
        self.metrics_logger.log_system_metrics()
        self.metrics_logger.logger.info(
            "Performance monitoring started",
            extra={'metric_type': 'monitoring', 'action': 'start'}
        )
    
    def checkpoint(self, name: str):
        """チェックポイントの記録"""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.checkpoints[name] = elapsed_time
        
        self.metrics_logger.logger.info(
            f"Performance checkpoint: {name}",
            extra={
                'metric_type': 'checkpoint',
                'checkpoint_name': name,
                'elapsed_time_seconds': round(elapsed_time, 3)
            }
        )
    
    def end_monitoring(self):
        """監視終了"""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
        
        total_time = time.time() - self.start_time
        self.metrics_logger.log_system_metrics()
        
        self.metrics_logger.logger.info(
            "Performance monitoring completed",
            extra={
                'metric_type': 'monitoring',
                'action': 'end',
                'total_execution_time_seconds': round(total_time, 3),
                'checkpoints': {k: round(v, 3) for k, v in self.checkpoints.items()}
            }
        )
        
        self.start_time = None
        self.checkpoints.clear()
        return total_time

def setup_application_logging(config: Optional[Config] = None):
    """アプリケーション全体のロギング設定"""
    if config is None:
        config = Config()
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # 既存のハンドラーをクリア
    root_logger.handlers.clear()
    
    # JSON形式のフォーマッター
    json_formatter = JSONFormatter()
    
    # ファイルハンドラー
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)
    
    # 開発環境でのコンソール出力
    if config.ENVIRONMENT == "development":
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 外部ライブラリのログレベルを調整
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info("Application logging configured successfully")