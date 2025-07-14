from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import timedelta
import os
import yaml

@dataclass
class Config:
    def __post_init__(self):
        """設定ファイルから値を読み込んで既定値を上書き"""
        self._load_from_yaml()
    
    def _load_from_yaml(self):
        """YAMLファイルから設定を読み込み"""
        yaml_path = Path(__file__).parent / "settings.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                # YAMLの設定で既定値を上書き
                self._update_from_dict(yaml_config)
                print(f"Configuration loaded from {yaml_path}")
            except Exception as e:
                print(f"Warning: Could not load YAML config: {e}")
                print("Using default configuration values")
        else:
            print(f"No YAML config found at {yaml_path}, using defaults")
    
    def _update_from_dict(self, config_dict: dict):
        """辞書から設定値を更新"""
        # 環境設定
        if 'environment' in config_dict:
            self.ENVIRONMENT = config_dict['environment']
        
        # データ取得設定
        if 'data_fetching' in config_dict:
            df = config_dict['data_fetching']
            if 'intraday_interval' in df:
                self.INTRADAY_INTERVAL = df['intraday_interval']
            if 'intraday_period_days' in df:
                self.INTRADAY_PERIOD_DAYS = df['intraday_period_days']
            if 'chart_longterm_period' in df:
                self.CHART_LONGTERM_PERIOD = df['chart_longterm_period']
            if 'target_calendar_countries' in df:
                self.TARGET_CALENDAR_COUNTRIES = df['target_calendar_countries']
        
        # Webスクレイピング設定
        if 'web_scraping' in config_dict:
            ws = config_dict['web_scraping']
            if 'user_agent' in ws:
                self.USER_AGENT_STRING = ws['user_agent']
            if 'webdriver_implicit_wait' in ws:
                self.WEBDRIVER_IMPLICIT_WAIT = ws['webdriver_implicit_wait']
            if 'webdriver_page_load_timeout' in ws:
                self.WEBDRIVER_PAGE_LOAD_TIMEOUT = ws['webdriver_page_load_timeout']
            if 'scraping_delay_seconds' in ws:
                self.SCRAPING_DELAY_SECONDS = ws['scraping_delay_seconds']
            if 'page_delay_seconds' in ws:
                self.PAGE_DELAY_SECONDS = ws['page_delay_seconds']
            if 'http_request_timeout' in ws:
                self.HTTP_REQUEST_TIMEOUT = ws['http_request_timeout']
        
        # Reuters設定
        if 'reuters' in config_dict:
            r = config_dict['reuters']
            if 'base_url' in r:
                self.REUTERS_BASE_URL = r['base_url']
            if 'search_url' in r:
                self.REUTERS_SEARCH_URL = r['search_url']
            if 'search_query' in r:
                self.REUTERS_SEARCH_QUERY = r['search_query']
            if 'target_categories' in r:
                self.REUTERS_TARGET_CATEGORIES = r['target_categories']
            if 'exclude_keywords' in r:
                self.REUTERS_EXCLUDE_KEYWORDS = r['exclude_keywords']
            if 'max_pages' in r:
                self.REUTERS_MAX_PAGES = r['max_pages']
        
        # AI設定
        if 'ai' in config_dict:
            ai = config_dict['ai']
            if 'preferred_models' in ai:
                self.GEMINI_PREFERRED_MODELS = ai['preferred_models']
            if 'text_limit' in ai:
                self.AI_TEXT_LIMIT = ai['text_limit']
        
        # ニュース設定
        if 'news' in config_dict:
            news = config_dict['news']
            if 'hours_limit' in news:
                self.NEWS_HOURS_LIMIT = news['hours_limit']
            if 'max_pages' in news:
                self.MAX_NEWS_PAGES = news['max_pages']
        
        # ファイル設定
        if 'files' in config_dict:
            files = config_dict['files']
            if 'report_filename' in files:
                self.REPORT_FILENAME = files['report_filename']
            if 'default_report_filename' in files:
                self.DEFAULT_REPORT_FILENAME = files['default_report_filename']
            if 'css_path' in files:
                self.CSS_PATH = files['css_path']
        
        # チャート設定
        if 'charts' in config_dict:
            charts = config_dict['charts']
            if 'width' in charts:
                self.CHART_WIDTH = charts['width']
            if 'height' in charts:
                self.CHART_HEIGHT = charts['height']
            if 'dpi' in charts:
                self.CHART_DPI = charts['dpi']
            if 'plotly_js_source' in charts:
                self.PLOTLY_JS_SOURCE = charts['plotly_js_source']
            if 'matplotlib_figure_size' in charts:
                self.MATPLOTLIB_FIGURE_SIZE = tuple(charts['matplotlib_figure_size'])
            if 'japanese_font_paths' in charts:
                self.JAPANESE_FONT_PATHS = charts['japanese_font_paths']
            if 'moving_averages' in charts:
                self.MOVING_AVERAGES = charts['moving_averages']
            if 'default_ma_display' in charts:
                self.DEFAULT_MA_DISPLAY = charts['default_ma_display']
            if 'default_ma_type' in charts:
                self.DEFAULT_MA_TYPE = charts['default_ma_type']
        
        # Markdown設定
        if 'markdown' in config_dict:
            markdown = config_dict['markdown']
            if 'extensions' in markdown:
                self.MARKDOWN_EXTENSIONS = markdown['extensions']
        
        # パフォーマンス設定
        if 'performance' in config_dict:
            perf = config_dict['performance']
            if 'max_workers' in perf:
                self.MAX_WORKERS = perf['max_workers']
            if 'timeout_seconds' in perf:
                self.TIMEOUT_SECONDS = perf['timeout_seconds']
            if 'retry_attempts' in perf:
                self.RETRY_ATTEMPTS = perf['retry_attempts']
            if 'retry_wait_min' in perf:
                self.RETRY_WAIT_MIN = perf['retry_wait_min']
            if 'retry_wait_max' in perf:
                self.RETRY_WAIT_MAX = perf['retry_wait_max']
        
        # ログ設定
        if 'logging' in config_dict:
            logging = config_dict['logging']
            if 'level' in logging:
                self.LOG_LEVEL = logging['level']
            if 'file' in logging:
                self.LOG_FILE = Path(__file__).parent / logging['file']
            if 'backup_count' in logging:
                self.LOG_BACKUP_COUNT = logging['backup_count']
            if 'max_bytes' in logging:
                self.LOG_MAX_BYTES = logging['max_bytes']
        
        # リソース制限
        if 'resources' in config_dict:
            res = config_dict['resources']
            if 'max_memory_mb' in res:
                self.MAX_MEMORY_MB = res['max_memory_mb']
            if 'max_disk_space_mb' in res:
                self.MAX_DISK_SPACE_MB = res['max_disk_space_mb']
        
        # マーケットティッカー
        if 'market_tickers' in config_dict:
            self.MARKET_TICKERS = config_dict['market_tickers']
        
        # セクターETF
        if 'sector_etfs' in config_dict:
            self.SECTOR_ETFS = config_dict['sector_etfs']
        
        # 資産分類
        if 'asset_classes' in config_dict:
            self.ASSET_CLASSES = config_dict['asset_classes']
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
    
    # 移動平均設定
    MOVING_AVERAGES: Dict[str, Dict] = field(default_factory=lambda: {
        "short": {"period": 25, "color": "blue", "label": "MA25"},
        "medium": {"period": 50, "color": "orange", "label": "MA50"}, 
        "long": {"period": 75, "color": "red", "label": "MA75"}
    })
    
    # デフォルトで表示する移動平均（キーのリスト）
    DEFAULT_MA_DISPLAY: List[str] = field(default_factory=lambda: ["short", "long"])
    
    # 移動平均タイプ設定
    MA_TYPES: Dict[str, str] = field(default_factory=lambda: {
        "SMA": "Simple Moving Average",
        "EMA": "Exponential Moving Average",
        "WMA": "Weighted Moving Average"
    })
    
    # デフォルト移動平均タイプ
    DEFAULT_MA_TYPE: str = "SMA"
    
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
    
    # マーケットデータ設定
    MARKET_TICKERS: Dict[str, str] = field(default_factory=lambda: {
        "S&P500": "^GSPC",
        "NASDAQ100": "^NDX",
        "ダウ30": "^DJI",
        "SOX": "^SOX",
        "日経225": "^N225",
        "米国2年金利": "^IRX",
        "米国10年金利": "^TNX",
        "VIX": "^VIX",
        "DXYドル指数": "DX-Y.NYB",
        "ドル円": "JPY=X",
        "ユーロドル": "EURUSD=X",
        "ビットコイン": "BTC-USD",
        "金": "GC=F",
        "原油": "CL=F"
    })
    
    # セクターETF設定
    SECTOR_ETFS: Dict[str, str] = field(default_factory=lambda: {
        "XLK": "Technology Select Sector SPDR Fund",
        "XLF": "Financial Select Sector SPDR Fund", 
        "XLY": "Consumer Discretionary Select Sector SPDR Fund",
        "XLP": "Consumer Staples Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund",
        "XLI": "Industrial Select Sector SPDR Fund",
        "XLB": "Materials Select Sector SPDR Fund",
        "XLRE": "Real Estate Select Sector SPDR Fund",
        "XLU": "Utilities Select Sector SPDR Fund",
        "XLC": "Communication Services Select Sector SPDR Fund"
    })
    
    # 資産分類設定
    ASSET_CLASSES: Dict[str, List[str]] = field(default_factory=lambda: {
        "US_STOCK": ["^GSPC", "^DJI", "^NDX", "^SOX", "^TNX", "^VIX"],
        "24H_ASSET": ["JPY=X", "EURUSD=X", "BTC-USD", "GC=F", "CL=F"]
    })
    
    # データ取得設定
    INTRADAY_INTERVAL: str = "5m"
    INTRADAY_PERIOD_DAYS: int = 7
    CHART_LONGTERM_PERIOD: str = "1y"
    TARGET_CALENDAR_COUNTRIES: List[str] = field(default_factory=lambda: ['united states'])
    
    # Webスクレイピング設定
    REUTERS_BASE_URL: str = "https://jp.reuters.com"
    REUTERS_SEARCH_URL: str = "https://jp.reuters.com/site-search/"
    USER_AGENT_STRING: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    WEBDRIVER_IMPLICIT_WAIT: int = 15
    WEBDRIVER_PAGE_LOAD_TIMEOUT: int = 120
    SCRAPING_DELAY_SECONDS: int = 7
    PAGE_DELAY_SECONDS: int = 1
    HTTP_REQUEST_TIMEOUT: int = 15
    
    # AI設定
    GEMINI_PREFERRED_MODELS: List[str] = field(default_factory=lambda: [
        'models/gemini-2.5-flash-lite-preview-06-17',
        'models/gemini-2.5-flash-preview-05-20'
    ])
    AI_TEXT_LIMIT: int = 1800
    
    # ニュース設定
    REUTERS_SEARCH_QUERY: str = "米国市場 OR 金融 OR 経済 OR 株価 OR FRB OR FOMC OR 決算 OR 利上げ OR インフレ"
    REUTERS_TARGET_CATEGORIES: List[str] = field(default_factory=lambda: [
        "ビジネスcategory", "マーケットcategory", "トップニュースcategory", 
        "ワールドcategory", "テクノロジーcategory", "アジア市場category",
        "不明", "経済category"
    ])
    REUTERS_EXCLUDE_KEYWORDS: List[str] = field(default_factory=lambda: [
        "スポーツ", "エンタメ", "五輪", "サッカー", "映画", "将棋", 
        "囲碁", "芸能", "ライフ", "アングル："
    ])
    REUTERS_MAX_PAGES: int = 5
    
    # ファイル設定
    REPORT_FILENAME: str = "index.html"
    DEFAULT_REPORT_FILENAME: str = "market_report.html"
    CSS_PATH: str = "static/style.css"
    
    # チャート詳細設定
    JAPANESE_FONT_PATHS: List[str] = field(default_factory=lambda: [
        '/System/Library/Fonts/ヒラギノ角ゴ ProN W3.ttc',
        '/System/Library/Fonts/Hiragino Sans/Hiragino Sans W3.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴ ProN W3.ttc',
        '/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/Library/Fonts/Osaka.ttf',
        '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
        '/System/Library/Fonts/ヒラギノ明朝 ProN W3.ttc'
    ])
    PLOTLY_JS_SOURCE: str = 'cdn'
    MATPLOTLIB_FIGURE_SIZE: tuple = (12, 6)
    CHART_DPI: int = 150
    
    # Markdown設定
    MARKDOWN_EXTENSIONS: List[str] = field(default_factory=lambda: ['extra', 'nl2br', 'sane_lists'])
    
    # 経済指標翻訳設定
    INDICATOR_TRANSLATIONS: Dict[str, str] = field(default_factory=lambda: {
        "Initial Jobless Claims": "新規失業保険申請件数",
        "Continuing Claims": "継続失業保険申請件数",
        "Nonfarm Payrolls": "非農業部門雇用者数",
        "Unemployment Rate": "失業率",
        "Consumer Price Index": "消費者物価指数",
        "Producer Price Index": "生産者物価指数",
        "Retail Sales": "小売売上高",
        "Industrial Production": "鉱工業生産",
        "Housing Starts": "住宅着工件数",
        "Building Permits": "建設許可件数",
        "Gross Domestic Product": "国内総生産",
        "Personal Income": "個人所得",
        "Personal Spending": "個人支出",
        "PCE Price Index": "PCE物価指数",
        "Durable Goods Orders": "耐久財受注",
        "Factory Orders": "製造業受注",
        "Business Inventories": "企業在庫",
        "Trade Balance": "貿易収支",
        "Current Account": "経常収支",
        "Import Price Index": "輸入物価指数",
        "Export Price Index": "輸出物価指数",
        "Treasury Budget": "財政収支",
        "Consumer Confidence": "消費者信頼感指数",
        "Consumer Sentiment": "消費者態度指数",
        "ISM Manufacturing PMI": "ISM製造業景況指数",
        "ISM Services PMI": "ISMサービス業景況指数",
        "New Home Sales": "新築住宅販売件数",
        "Existing Home Sales": "既存住宅販売件数",
        "Pending Home Sales": "住宅販売仮契約件数",
        "Case-Shiller Home Price Index": "ケース・シラー住宅価格指数",
        "FHFA Home Price Index": "FHFA住宅価格指数",
        "MBA Mortgage Applications": "MBA住宅ローン申請件数",
        "Mortgage Rates": "住宅ローン金利",
        "Chicago PMI": "シカゴ購買部協会景気指数",
        "Philadelphia Fed Index": "フィラデルフィア連銀製造業景況指数",
        "New York Fed Index": "ニューヨーク連銀製造業景況指数",
        "Kansas City Fed Index": "カンザスシティ連銀製造業景況指数",
        "Richmond Fed Index": "リッチモンド連銀製造業景況指数",
        "Dallas Fed Index": "ダラス連銀製造業景況指数"
    })
