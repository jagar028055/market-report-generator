from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import timedelta
import os

@dataclass
class Config:
    # 環境設定
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # development/production
    
    # ディレクトリ設定
    BASE_DIR: Path = Path(__file__).parent.resolve()
    CHARTS_DIR: Path = BASE_DIR / "charts"
    TEMPLATE_DIR: Path = BASE_DIR / "templates"
    OUTPUT_DIR: Path = BASE_DIR
    BACKUP_DIR: Path = BASE_DIR / "backup"
    
    # パフォーマンス設定
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 30
    RETRY_ATTEMPTS: int = 3
    RETRY_WAIT_MIN: int = 4
    RETRY_WAIT_MAX: int = 10
    
    # データ取得設定
    NEWS_HOURS_LIMIT: int = 24
    MAX_NEWS_PAGES: int = 5
    
    # ログ設定
    LOG_LEVEL: str = "INFO" if ENVIRONMENT == "production" else "DEBUG"
    LOG_FILE: Path = BASE_DIR / "execution.log"
    LOG_BACKUP_COUNT: int = 7  # 1週間分のバックアップを保持
    LOG_MAX_BYTES: int = 10485760  # 10MB
    
    # チャート設定
    CHART_WIDTH: int = 1200
    CHART_HEIGHT: int = 600
    
    # リソース制限
    MAX_MEMORY_MB: int = 512
    MAX_DISK_SPACE_MB: int = 1024
    
    # タイムアウト設定
    API_TIMEOUT: timedelta = timedelta(seconds=30)
    
    # データキャッシュ設定
    CACHE_ENABLED: bool = True
    CACHE_TTL: timedelta = timedelta(hours=24)
    
    # メトリクス設定
    METRICS_ENABLED: bool = True
    METRICS_INTERVAL: timedelta = timedelta(minutes=5)
    
    # エラーハンドリング設定
    MAX_RETRIES: int = 3
    RETRY_DELAY: timedelta = timedelta(seconds=5)
    
    # テスト設定
    TEST_MODE: bool = False
    TEST_DATA_DIR: Path = BASE_DIR / "tests" / "data"
    
    # データバックアップ設定
    BACKUP_INTERVAL: timedelta = timedelta(days=1)
    MAX_BACKUP_DAYS: int = 7
    
    # データ圧縮設定
    COMPRESS_DATA: bool = True
    COMPRESS_LEVEL: int = 9
    
    # エラーメール通知設定
    ERROR_EMAIL_ENABLED: bool = True if ENVIRONMENT == "production" else False
    ERROR_EMAIL_TO: Optional[str] = None  # 環境変数から取得
    ERROR_EMAIL_FROM: Optional[str] = None  # 環境変数から取得
    
    # データクリーンアップ設定
    CLEANUP_INTERVAL: timedelta = timedelta(days=7)
    MAX_DATA_AGE: timedelta = timedelta(days=30)
