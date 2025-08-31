"""
Market Report Generator Configuration
設定値の管理とデフォルト値の定義
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """アプリケーション設定クラス"""
    
    def __init__(self):
        # ベースディレクトリの設定
        self.BASE_DIR = Path(__file__).parent.parent
        
        # 環境設定
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        
        # ログ設定
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", str(self.BASE_DIR / "logs" / "market_report.log"))
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
        self.LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
        # API設定
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_DOCS_ID = os.getenv("GOOGLE_DOCS_ID")
        
        # Web UI設定
        self.WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
        self.WEB_PORT = int(os.getenv("WEB_PORT", "5000"))
        self.WEB_DEBUG = os.getenv("WEB_DEBUG", "true").lower() == "true"
        
        # データ取得設定
        self.DATA_UPDATE_INTERVAL = int(os.getenv("DATA_UPDATE_INTERVAL", "300"))  # 5分
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # チャート設定
        self.CHART_WIDTH = int(os.getenv("CHART_WIDTH", "1200"))
        self.CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", "600"))
        
        # ディレクトリの作成
        self._ensure_directories()
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        directories = [
            Path(self.LOG_FILE).parent,
            self.BASE_DIR / "charts",
            self.BASE_DIR / "static",
            self.BASE_DIR / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """開発環境かどうかを判定"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """本番環境かどうかを判定"""
        return self.ENVIRONMENT.lower() == "production"
    
    def get_api_key(self, service: str) -> Optional[str]:
        """APIキーを取得"""
        key_mapping = {
            "openai": self.OPENAI_API_KEY,
            "alpha_vantage": self.ALPHA_VANTAGE_API_KEY,
            "google": self.GOOGLE_API_KEY
        }
        return key_mapping.get(service.lower())
    
    def validate_config(self) -> list:
        """設定の検証を行い、問題があれば警告リストを返す"""
        warnings = []
        
        if not self.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - AI commentary will be disabled")
        
        if not self.ALPHA_VANTAGE_API_KEY:
            warnings.append("ALPHA_VANTAGE_API_KEY not set - some data sources may be limited")
        
        if not self.GOOGLE_DOCS_ID:
            warnings.append("GOOGLE_DOCS_ID not set - Google Docs news source will be disabled")
        
        return warnings


def get_system_config() -> Config:
    """システム設定のシングルトンインスタンスを取得"""
    if not hasattr(get_system_config, '_config'):
        get_system_config._config = Config()
    return get_system_config._config


# デフォルト設定インスタンス
default_config = Config()