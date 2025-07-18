"""
システム関連の設定
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import timedelta

from .base_config import BaseConfig, ConfigValidator


@dataclass
class SystemConfig(BaseConfig):
    """システム設定"""
    
    # 環境設定
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # ディレクトリ設定
    BASE_DIR: Path = Path(__file__).parent.parent.parent.resolve()
    CHARTS_DIR: Path = BASE_DIR / "charts"
    TEMPLATE_DIR: Path = BASE_DIR / "templates"
    OUTPUT_DIR: Path = BASE_DIR
    BACKUP_DIR: Path = BASE_DIR / "backup"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    # パフォーマンス設定
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 30
    RETRY_ATTEMPTS: int = 3
    RETRY_WAIT_MIN: int = 1
    RETRY_WAIT_MAX: int = 60
    
    # リソース制限
    MAX_MEMORY_MB: int = 512
    MAX_DISK_SPACE_MB: int = 1024
    
    # 並行処理設定
    ENABLE_PARALLEL_PROCESSING: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # セキュリティ設定
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_ERROR_TRACKING: bool = True
    SENSITIVE_DATA_FIELDS: List[str] = field(default_factory=lambda: [
        "api_key", "password", "token", "secret", "credentials"
    ])
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'system' in config_dict:
            system_config = config_dict['system']
            
            # 環境設定
            if 'environment' in system_config:
                self.ENVIRONMENT = system_config['environment']
            
            # パフォーマンス設定
            if 'performance' in system_config:
                perf_config = system_config['performance']
                if 'max_workers' in perf_config:
                    self.MAX_WORKERS = perf_config['max_workers']
                if 'timeout_seconds' in perf_config:
                    self.TIMEOUT_SECONDS = perf_config['timeout_seconds']
                if 'retry_attempts' in perf_config:
                    self.RETRY_ATTEMPTS = perf_config['retry_attempts']
                if 'retry_wait_min' in perf_config:
                    self.RETRY_WAIT_MIN = perf_config['retry_wait_min']
                if 'retry_wait_max' in perf_config:
                    self.RETRY_WAIT_MAX = perf_config['retry_wait_max']
            
            # リソース設定
            if 'resources' in system_config:
                res_config = system_config['resources']
                if 'max_memory_mb' in res_config:
                    self.MAX_MEMORY_MB = res_config['max_memory_mb']
                if 'max_disk_space_mb' in res_config:
                    self.MAX_DISK_SPACE_MB = res_config['max_disk_space_mb']
            
            # 並行処理設定
            if 'enable_parallel_processing' in system_config:
                self.ENABLE_PARALLEL_PROCESSING = system_config['enable_parallel_processing']
            if 'max_concurrent_requests' in system_config:
                self.MAX_CONCURRENT_REQUESTS = system_config['max_concurrent_requests']
            
            # セキュリティ設定
            if 'security' in system_config:
                sec_config = system_config['security']
                if 'enable_request_logging' in sec_config:
                    self.ENABLE_REQUEST_LOGGING = sec_config['enable_request_logging']
                if 'enable_error_tracking' in sec_config:
                    self.ENABLE_ERROR_TRACKING = sec_config['enable_error_tracking']
                if 'sensitive_data_fields' in sec_config:
                    self.SENSITIVE_DATA_FIELDS = sec_config['sensitive_data_fields']
    
    def _validate_configuration(self):
        """設定値の検証"""
        # 環境設定の検証
        ConfigValidator.validate_choice(
            self.ENVIRONMENT, 
            ["development", "staging", "production"], 
            "ENVIRONMENT"
        )
        
        # パフォーマンス設定の検証
        ConfigValidator.validate_positive_integer(self.MAX_WORKERS, "MAX_WORKERS")
        ConfigValidator.validate_positive_integer(self.TIMEOUT_SECONDS, "TIMEOUT_SECONDS")
        ConfigValidator.validate_positive_integer(self.RETRY_ATTEMPTS, "RETRY_ATTEMPTS")
        ConfigValidator.validate_positive_integer(self.RETRY_WAIT_MIN, "RETRY_WAIT_MIN")
        ConfigValidator.validate_positive_integer(self.RETRY_WAIT_MAX, "RETRY_WAIT_MAX")
        
        # リソース設定の検証
        ConfigValidator.validate_positive_integer(self.MAX_MEMORY_MB, "MAX_MEMORY_MB")
        ConfigValidator.validate_positive_integer(self.MAX_DISK_SPACE_MB, "MAX_DISK_SPACE_MB")
        ConfigValidator.validate_positive_integer(self.MAX_CONCURRENT_REQUESTS, "MAX_CONCURRENT_REQUESTS")
        
        # 論理設定の検証
        if not isinstance(self.ENABLE_PARALLEL_PROCESSING, bool):
            raise ValueError("ENABLE_PARALLEL_PROCESSING must be a boolean")
        if not isinstance(self.ENABLE_REQUEST_LOGGING, bool):
            raise ValueError("ENABLE_REQUEST_LOGGING must be a boolean")
        if not isinstance(self.ENABLE_ERROR_TRACKING, bool):
            raise ValueError("ENABLE_ERROR_TRACKING must be a boolean")
        
        # リスト設定の検証
        ConfigValidator.validate_list(self.SENSITIVE_DATA_FIELDS, "SENSITIVE_DATA_FIELDS")
        
        # 待機時間の論理的検証
        if self.RETRY_WAIT_MIN >= self.RETRY_WAIT_MAX:
            raise ValueError("RETRY_WAIT_MIN must be less than RETRY_WAIT_MAX")
    
    def create_directories(self):
        """必要なディレクトリを作成"""
        directories = [
            self.CHARTS_DIR,
            self.BACKUP_DIR,
            self.LOGS_DIR,
            self.CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """本番環境かどうかを判定"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """開発環境かどうかを判定"""
        return self.ENVIRONMENT == "development"


@dataclass
class LoggingConfig(BaseConfig):
    """ログ設定"""
    
    # ログレベル設定
    LOG_LEVEL: str = "INFO"
    
    # ログファイル設定
    LOG_FILE: str = "execution.log"
    ERROR_LOG_FILE: str = "error.log"
    ACCESS_LOG_FILE: str = "access.log"
    
    # ログローテーション設定
    LOG_BACKUP_COUNT: int = 7
    LOG_MAX_BYTES: int = 10485760  # 10MB
    
    # ログフォーマット設定
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # 構造化ログ設定
    ENABLE_STRUCTURED_LOGGING: bool = False
    LOG_OUTPUT_FORMAT: str = "text"  # text, json
    
    # パフォーマンスログ設定
    ENABLE_PERFORMANCE_LOGGING: bool = True
    PERFORMANCE_LOG_THRESHOLD: float = 1.0  # 秒
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'logging' in config_dict:
            logging_config = config_dict['logging']
            
            # ログレベル設定
            if 'level' in logging_config:
                self.LOG_LEVEL = logging_config['level']
            
            # ログファイル設定
            if 'log_file' in logging_config:
                self.LOG_FILE = logging_config['log_file']
            if 'error_log_file' in logging_config:
                self.ERROR_LOG_FILE = logging_config['error_log_file']
            if 'access_log_file' in logging_config:
                self.ACCESS_LOG_FILE = logging_config['access_log_file']
            
            # ローテーション設定
            if 'backup_count' in logging_config:
                self.LOG_BACKUP_COUNT = logging_config['backup_count']
            if 'max_bytes' in logging_config:
                self.LOG_MAX_BYTES = logging_config['max_bytes']
            
            # フォーマット設定
            if 'format' in logging_config:
                self.LOG_FORMAT = logging_config['format']
            if 'date_format' in logging_config:
                self.DATE_FORMAT = logging_config['date_format']
            
            # 構造化ログ設定
            if 'enable_structured_logging' in logging_config:
                self.ENABLE_STRUCTURED_LOGGING = logging_config['enable_structured_logging']
            if 'output_format' in logging_config:
                self.LOG_OUTPUT_FORMAT = logging_config['output_format']
            
            # パフォーマンスログ設定
            if 'enable_performance_logging' in logging_config:
                self.ENABLE_PERFORMANCE_LOGGING = logging_config['enable_performance_logging']
            if 'performance_log_threshold' in logging_config:
                self.PERFORMANCE_LOG_THRESHOLD = logging_config['performance_log_threshold']
    
    def _validate_configuration(self):
        """設定値の検証"""
        # ログレベルの検証
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ConfigValidator.validate_choice(self.LOG_LEVEL, valid_levels, "LOG_LEVEL")
        
        # ファイル名の検証
        ConfigValidator.validate_string(self.LOG_FILE, "LOG_FILE")
        ConfigValidator.validate_string(self.ERROR_LOG_FILE, "ERROR_LOG_FILE")
        ConfigValidator.validate_string(self.ACCESS_LOG_FILE, "ACCESS_LOG_FILE")
        
        # ローテーション設定の検証
        ConfigValidator.validate_positive_integer(self.LOG_BACKUP_COUNT, "LOG_BACKUP_COUNT")
        ConfigValidator.validate_positive_integer(self.LOG_MAX_BYTES, "LOG_MAX_BYTES")
        
        # フォーマット設定の検証
        ConfigValidator.validate_string(self.LOG_FORMAT, "LOG_FORMAT")
        ConfigValidator.validate_string(self.DATE_FORMAT, "DATE_FORMAT")
        
        # 出力フォーマットの検証
        ConfigValidator.validate_choice(
            self.LOG_OUTPUT_FORMAT, 
            ["text", "json"], 
            "LOG_OUTPUT_FORMAT"
        )
        
        # パフォーマンス設定の検証
        ConfigValidator.validate_positive_float(
            self.PERFORMANCE_LOG_THRESHOLD, 
            "PERFORMANCE_LOG_THRESHOLD"
        )
        
        # 論理設定の検証
        if not isinstance(self.ENABLE_STRUCTURED_LOGGING, bool):
            raise ValueError("ENABLE_STRUCTURED_LOGGING must be a boolean")
        if not isinstance(self.ENABLE_PERFORMANCE_LOGGING, bool):
            raise ValueError("ENABLE_PERFORMANCE_LOGGING must be a boolean")


@dataclass
class FileConfig(BaseConfig):
    """ファイル関連設定"""
    
    # レポートファイル設定
    REPORT_FILENAME: str = "index.html"
    DEFAULT_REPORT_FILENAME: str = "market_report.html"
    
    # スタイルファイル設定
    CSS_PATH: str = "static/style.css"
    
    # バックアップ設定
    BACKUP_INTERVAL: timedelta = timedelta(days=1)
    MAX_BACKUP_DAYS: int = 7
    ENABLE_AUTO_BACKUP: bool = True
    
    # ファイル圧縮設定
    COMPRESS_DATA: bool = True
    COMPRESS_LEVEL: int = 9
    
    # 一時ファイル設定
    TEMP_DIR: str = "temp"
    CLEANUP_TEMP_FILES: bool = True
    TEMP_FILE_MAX_AGE: timedelta = timedelta(hours=24)
    
    # ファイルアクセス設定
    DEFAULT_FILE_PERMISSIONS: int = 0o644
    DEFAULT_DIR_PERMISSIONS: int = 0o755
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'files' in config_dict:
            file_config = config_dict['files']
            
            # レポートファイル設定
            if 'report_filename' in file_config:
                self.REPORT_FILENAME = file_config['report_filename']
            if 'default_report_filename' in file_config:
                self.DEFAULT_REPORT_FILENAME = file_config['default_report_filename']
            if 'css_path' in file_config:
                self.CSS_PATH = file_config['css_path']
            
            # バックアップ設定
            if 'backup' in file_config:
                backup_config = file_config['backup']
                if 'interval_days' in backup_config:
                    self.BACKUP_INTERVAL = timedelta(days=backup_config['interval_days'])
                if 'max_backup_days' in backup_config:
                    self.MAX_BACKUP_DAYS = backup_config['max_backup_days']
                if 'enable_auto_backup' in backup_config:
                    self.ENABLE_AUTO_BACKUP = backup_config['enable_auto_backup']
            
            # 圧縮設定
            if 'compress_data' in file_config:
                self.COMPRESS_DATA = file_config['compress_data']
            if 'compress_level' in file_config:
                self.COMPRESS_LEVEL = file_config['compress_level']
            
            # 一時ファイル設定
            if 'temp_dir' in file_config:
                self.TEMP_DIR = file_config['temp_dir']
            if 'cleanup_temp_files' in file_config:
                self.CLEANUP_TEMP_FILES = file_config['cleanup_temp_files']
            if 'temp_file_max_age_hours' in file_config:
                self.TEMP_FILE_MAX_AGE = timedelta(hours=file_config['temp_file_max_age_hours'])
    
    def _validate_configuration(self):
        """設定値の検証"""
        # ファイル名の検証
        ConfigValidator.validate_string(self.REPORT_FILENAME, "REPORT_FILENAME")
        ConfigValidator.validate_string(self.DEFAULT_REPORT_FILENAME, "DEFAULT_REPORT_FILENAME")
        ConfigValidator.validate_string(self.CSS_PATH, "CSS_PATH")
        ConfigValidator.validate_string(self.TEMP_DIR, "TEMP_DIR")
        
        # バックアップ設定の検証
        ConfigValidator.validate_positive_integer(self.MAX_BACKUP_DAYS, "MAX_BACKUP_DAYS")
        
        # 圧縮設定の検証
        if not isinstance(self.COMPRESS_LEVEL, int) or not (1 <= self.COMPRESS_LEVEL <= 9):
            raise ValueError("COMPRESS_LEVEL must be an integer between 1 and 9")
        
        # 論理設定の検証
        if not isinstance(self.COMPRESS_DATA, bool):
            raise ValueError("COMPRESS_DATA must be a boolean")
        if not isinstance(self.ENABLE_AUTO_BACKUP, bool):
            raise ValueError("ENABLE_AUTO_BACKUP must be a boolean")
        if not isinstance(self.CLEANUP_TEMP_FILES, bool):
            raise ValueError("CLEANUP_TEMP_FILES must be a boolean")
        
        # パーミッション設定の検証
        if not isinstance(self.DEFAULT_FILE_PERMISSIONS, int):
            raise ValueError("DEFAULT_FILE_PERMISSIONS must be an integer")
        if not isinstance(self.DEFAULT_DIR_PERMISSIONS, int):
            raise ValueError("DEFAULT_DIR_PERMISSIONS must be an integer")
    
    def get_report_path(self, base_dir: Path = None) -> Path:
        """レポートファイルのパスを取得"""
        if base_dir is None:
            base_dir = Path.cwd()
        return base_dir / self.REPORT_FILENAME
    
    def get_css_path(self, base_dir: Path = None) -> Path:
        """CSSファイルのパスを取得"""
        if base_dir is None:
            base_dir = Path.cwd()
        return base_dir / self.CSS_PATH
    
    def get_temp_dir(self, base_dir: Path = None) -> Path:
        """一時ディレクトリのパスを取得"""
        if base_dir is None:
            base_dir = Path.cwd()
        return base_dir / self.TEMP_DIR