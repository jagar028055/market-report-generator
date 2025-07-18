"""
設定管理モジュール
"""

from .base_config import BaseConfig, ConfigValidator, ConfigManager, get_config_manager
from .data_config import DataFetchConfig, NewsConfig, WebScrapingConfig
from .chart_config import ChartConfig, AIConfig
from .system_config import SystemConfig, LoggingConfig, FileConfig


class AppConfig:
    """アプリケーション全体の設定を管理するクラス"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        
        # 各設定を初期化
        self.data_fetch = DataFetchConfig()
        self.news = NewsConfig()
        self.web_scraping = WebScrapingConfig()
        self.chart = ChartConfig()
        self.ai = AIConfig()
        self.system = SystemConfig()
        self.logging = LoggingConfig()
        self.file = FileConfig()
        
        # 設定マネージャーに登録
        self.config_manager.register_config("data_fetch", self.data_fetch)
        self.config_manager.register_config("news", self.news)
        self.config_manager.register_config("web_scraping", self.web_scraping)
        self.config_manager.register_config("chart", self.chart)
        self.config_manager.register_config("ai", self.ai)
        self.config_manager.register_config("system", self.system)
        self.config_manager.register_config("logging", self.logging)
        self.config_manager.register_config("file", self.file)
        
        # 必要なディレクトリを作成
        self.system.create_directories()
    
    def validate_all(self):
        """すべての設定を検証"""
        self.config_manager.validate_all_configs()
    
    def reload_all(self):
        """すべての設定を再読み込み"""
        self.config_manager.reload_all_configs()
    
    def get_config(self, name: str) -> BaseConfig:
        """指定された設定を取得"""
        return self.config_manager.get_config(name)
    
    def to_dict(self) -> dict:
        """全設定を辞書形式で取得"""
        return {
            name: config.to_dict() 
            for name, config in self.config_manager.get_all_configs().items()
        }


# シングルトンパターンでアプリケーション設定を提供
_app_config = None

def get_app_config() -> AppConfig:
    """アプリケーション設定のシングルトンインスタンスを取得"""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


# 便利な関数
def get_data_config() -> DataFetchConfig:
    """データ取得設定を取得"""
    return get_app_config().data_fetch


def get_news_config() -> NewsConfig:
    """ニュース設定を取得"""
    return get_app_config().news


def get_web_scraping_config() -> WebScrapingConfig:
    """Webスクレイピング設定を取得"""
    return get_app_config().web_scraping


def get_chart_config() -> ChartConfig:
    """チャート設定を取得"""
    return get_app_config().chart


def get_ai_config() -> AIConfig:
    """AI設定を取得"""
    return get_app_config().ai


def get_system_config() -> SystemConfig:
    """システム設定を取得"""
    return get_app_config().system


def get_logging_config() -> LoggingConfig:
    """ログ設定を取得"""
    return get_app_config().logging


def get_file_config() -> FileConfig:
    """ファイル設定を取得"""
    return get_app_config().file


# 下位互換性のための旧設定クラス
class Config:
    """下位互換性のための旧設定クラス"""
    
    def __init__(self):
        self.app_config = get_app_config()
        
        # 旧設定への参照を設定
        self._setup_legacy_attributes()
    
    def _setup_legacy_attributes(self):
        """旧設定属性を設定"""
        # データ取得設定
        data_config = self.app_config.data_fetch
        self.MARKET_TICKERS = data_config.MARKET_TICKERS
        self.SECTOR_ETFS = data_config.SECTOR_ETFS
        self.ASSET_CLASSES = data_config.ASSET_CLASSES
        self.INTRADAY_INTERVAL = data_config.INTRADAY_INTERVAL
        self.INTRADAY_PERIOD_DAYS = data_config.INTRADAY_PERIOD_DAYS
        self.CHART_LONGTERM_PERIOD = data_config.CHART_LONGTERM_PERIOD
        self.TARGET_CALENDAR_COUNTRIES = data_config.TARGET_CALENDAR_COUNTRIES
        self.INDICATOR_TRANSLATIONS = data_config.INDICATOR_TRANSLATIONS
        
        # ニュース設定
        news_config = self.app_config.news
        self.NEWS_HOURS_LIMIT = news_config.NEWS_HOURS_LIMIT
        self.MAX_NEWS_PAGES = news_config.MAX_NEWS_PAGES
        self.REUTERS_BASE_URL = news_config.REUTERS_BASE_URL
        self.REUTERS_SEARCH_URL = news_config.REUTERS_SEARCH_URL
        self.REUTERS_SEARCH_QUERY = news_config.REUTERS_SEARCH_QUERY
        self.REUTERS_TARGET_CATEGORIES = news_config.REUTERS_TARGET_CATEGORIES
        self.REUTERS_EXCLUDE_KEYWORDS = news_config.REUTERS_EXCLUDE_KEYWORDS
        self.REUTERS_MAX_PAGES = news_config.REUTERS_MAX_PAGES
        
        # Webスクレイピング設定
        web_config = self.app_config.web_scraping
        self.USER_AGENT_STRING = web_config.USER_AGENT_STRING
        self.WEBDRIVER_IMPLICIT_WAIT = web_config.WEBDRIVER_IMPLICIT_WAIT
        self.WEBDRIVER_PAGE_LOAD_TIMEOUT = web_config.WEBDRIVER_PAGE_LOAD_TIMEOUT
        self.SCRAPING_DELAY_SECONDS = web_config.SCRAPING_DELAY_SECONDS
        self.PAGE_DELAY_SECONDS = web_config.PAGE_DELAY_SECONDS
        self.HTTP_REQUEST_TIMEOUT = web_config.HTTP_REQUEST_TIMEOUT
        
        # チャート設定
        chart_config = self.app_config.chart
        self.CHART_WIDTH = chart_config.CHART_WIDTH
        self.CHART_HEIGHT = chart_config.CHART_HEIGHT
        self.CHART_DPI = chart_config.CHART_DPI
        self.JAPANESE_FONT_PATHS = chart_config.JAPANESE_FONT_PATHS
        self.MOVING_AVERAGES = chart_config.MOVING_AVERAGES
        self.DEFAULT_MA_DISPLAY = chart_config.DEFAULT_MA_DISPLAY
        self.DEFAULT_MA_TYPE = chart_config.DEFAULT_MA_TYPE
        self.PLOTLY_JS_SOURCE = chart_config.PLOTLY_JS_SOURCE
        self.MATPLOTLIB_FIGURE_SIZE = chart_config.MATPLOTLIB_FIGURE_SIZE
        
        # AI設定
        ai_config = self.app_config.ai
        self.GEMINI_PREFERRED_MODELS = ai_config.GEMINI_PREFERRED_MODELS
        self.AI_TEXT_LIMIT = ai_config.AI_TEXT_LIMIT
        
        # システム設定
        system_config = self.app_config.system
        self.ENVIRONMENT = system_config.ENVIRONMENT
        self.BASE_DIR = system_config.BASE_DIR
        self.CHARTS_DIR = system_config.CHARTS_DIR
        self.TEMPLATE_DIR = system_config.TEMPLATE_DIR
        self.OUTPUT_DIR = system_config.OUTPUT_DIR
        self.BACKUP_DIR = system_config.BACKUP_DIR
        self.MAX_WORKERS = system_config.MAX_WORKERS
        self.TIMEOUT_SECONDS = system_config.TIMEOUT_SECONDS
        self.RETRY_ATTEMPTS = system_config.RETRY_ATTEMPTS
        self.RETRY_WAIT_MIN = system_config.RETRY_WAIT_MIN
        self.RETRY_WAIT_MAX = system_config.RETRY_WAIT_MAX
        self.MAX_MEMORY_MB = system_config.MAX_MEMORY_MB
        self.MAX_DISK_SPACE_MB = system_config.MAX_DISK_SPACE_MB
        
        # ログ設定
        logging_config = self.app_config.logging
        self.LOG_LEVEL = logging_config.LOG_LEVEL
        self.LOG_FILE = logging_config.LOG_FILE
        self.LOG_BACKUP_COUNT = logging_config.LOG_BACKUP_COUNT
        self.LOG_MAX_BYTES = logging_config.LOG_MAX_BYTES
        
        # ファイル設定
        file_config = self.app_config.file
        self.REPORT_FILENAME = file_config.REPORT_FILENAME
        self.DEFAULT_REPORT_FILENAME = file_config.DEFAULT_REPORT_FILENAME
        self.CSS_PATH = file_config.CSS_PATH


# 下位互換性のためのエクスポート
__all__ = [
    'AppConfig',
    'Config',
    'get_app_config',
    'get_data_config',
    'get_news_config',
    'get_web_scraping_config',
    'get_chart_config',
    'get_ai_config',
    'get_system_config',
    'get_logging_config',
    'get_file_config',
    'BaseConfig',
    'ConfigValidator',
    'ConfigManager',
    'DataFetchConfig',
    'NewsConfig',
    'WebScrapingConfig',
    'ChartConfig',
    'AIConfig',
    'SystemConfig',
    'LoggingConfig',
    'FileConfig'
]