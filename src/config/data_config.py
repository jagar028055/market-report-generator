"""
データ取得関連の設定
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path

from .base_config import BaseConfig, ConfigValidator


@dataclass
class DataFetchConfig(BaseConfig):
    """データ取得設定"""
    
    # 市場データ設定
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
        "ゴールド": "GC=F",
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
    
    # イントラデイデータ設定
    INTRADAY_INTERVAL: str = "5m"
    INTRADAY_PERIOD_DAYS: int = 7
    
    # 長期データ設定
    CHART_LONGTERM_PERIOD: str = "1y"
    
    # 経済指標設定
    TARGET_CALENDAR_COUNTRIES: List[str] = field(default_factory=lambda: ['united states'])
    
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
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'data_fetching' in config_dict:
            data_config = config_dict['data_fetching']
            
            # 基本設定
            if 'intraday_interval' in data_config:
                self.INTRADAY_INTERVAL = data_config['intraday_interval']
            if 'intraday_period_days' in data_config:
                self.INTRADAY_PERIOD_DAYS = data_config['intraday_period_days']
            if 'chart_longterm_period' in data_config:
                self.CHART_LONGTERM_PERIOD = data_config['chart_longterm_period']
            if 'target_calendar_countries' in data_config:
                self.TARGET_CALENDAR_COUNTRIES = data_config['target_calendar_countries']
            
            # ティッカー設定
            if 'market_tickers' in data_config:
                self.MARKET_TICKERS.update(data_config['market_tickers'])
            if 'sector_etfs' in data_config:
                self.SECTOR_ETFS.update(data_config['sector_etfs'])
            if 'asset_classes' in data_config:
                self.ASSET_CLASSES.update(data_config['asset_classes'])
            
            # 翻訳設定
            if 'indicator_translations' in data_config:
                self.INDICATOR_TRANSLATIONS.update(data_config['indicator_translations'])
    
    def _validate_configuration(self):
        """設定値の検証"""
        # 基本設定の検証
        ConfigValidator.validate_string(self.INTRADAY_INTERVAL, "INTRADAY_INTERVAL")
        ConfigValidator.validate_positive_integer(self.INTRADAY_PERIOD_DAYS, "INTRADAY_PERIOD_DAYS")
        ConfigValidator.validate_string(self.CHART_LONGTERM_PERIOD, "CHART_LONGTERM_PERIOD")
        
        # 辞書設定の検証
        ConfigValidator.validate_dict(self.MARKET_TICKERS, "MARKET_TICKERS", min_items=1)
        ConfigValidator.validate_dict(self.SECTOR_ETFS, "SECTOR_ETFS", min_items=1)
        ConfigValidator.validate_dict(self.ASSET_CLASSES, "ASSET_CLASSES", min_items=1)
        ConfigValidator.validate_dict(self.INDICATOR_TRANSLATIONS, "INDICATOR_TRANSLATIONS")
        
        # リスト設定の検証
        ConfigValidator.validate_list(self.TARGET_CALENDAR_COUNTRIES, "TARGET_CALENDAR_COUNTRIES", min_items=1)
        
        # 資産分類の詳細検証
        for asset_class, symbols in self.ASSET_CLASSES.items():
            if not isinstance(symbols, list):
                raise ValueError(f"ASSET_CLASSES['{asset_class}'] must be a list")
            if not symbols:
                raise ValueError(f"ASSET_CLASSES['{asset_class}'] must not be empty")
    
    def add_ticker(self, name: str, symbol: str):
        """ティッカーを追加"""
        self.MARKET_TICKERS[name] = symbol
    
    def remove_ticker(self, name: str):
        """ティッカーを削除"""
        if name in self.MARKET_TICKERS:
            del self.MARKET_TICKERS[name]
    
    def add_sector_etf(self, symbol: str, name: str):
        """セクターETFを追加"""
        self.SECTOR_ETFS[symbol] = name
    
    def remove_sector_etf(self, symbol: str):
        """セクターETFを削除"""
        if symbol in self.SECTOR_ETFS:
            del self.SECTOR_ETFS[symbol]
    
    def add_translation(self, english_name: str, japanese_name: str):
        """指標翻訳を追加"""
        self.INDICATOR_TRANSLATIONS[english_name] = japanese_name
    
    def get_ticker_symbol(self, name: str) -> str:
        """ティッカー名からシンボルを取得"""
        return self.MARKET_TICKERS.get(name, name)
    
    def get_japanese_indicator_name(self, english_name: str) -> str:
        """英語の指標名から日本語名を取得"""
        return self.INDICATOR_TRANSLATIONS.get(english_name, english_name)
    
    def is_us_stock(self, symbol: str) -> bool:
        """米国株式かどうかを判定"""
        return symbol in self.ASSET_CLASSES.get("US_STOCK", [])
    
    def is_24h_asset(self, symbol: str) -> bool:
        """24時間取引資産かどうかを判定"""
        return symbol in self.ASSET_CLASSES.get("24H_ASSET", [])


@dataclass
class NewsConfig(BaseConfig):
    """ニュース取得設定"""
    
    # 基本設定
    NEWS_HOURS_LIMIT: int = 24
    MAX_NEWS_PAGES: int = 5
    
    # Reuters設定
    REUTERS_BASE_URL: str = "https://jp.reuters.com"
    REUTERS_SEARCH_URL: str = "https://jp.reuters.com/site-search/"
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
    
    # Google Docs設定
    GOOGLE_DOCS_ENABLED: bool = True
    GOOGLE_DOCS_CREDENTIALS_PATH: str = "service_account.json"
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'news' in config_dict:
            news_config = config_dict['news']
            
            # 基本設定
            if 'hours_limit' in news_config:
                self.NEWS_HOURS_LIMIT = news_config['hours_limit']
            if 'max_pages' in news_config:
                self.MAX_NEWS_PAGES = news_config['max_pages']
            
            # Reuters設定
            if 'reuters' in news_config:
                reuters_config = news_config['reuters']
                if 'base_url' in reuters_config:
                    self.REUTERS_BASE_URL = reuters_config['base_url']
                if 'search_url' in reuters_config:
                    self.REUTERS_SEARCH_URL = reuters_config['search_url']
                if 'search_query' in reuters_config:
                    self.REUTERS_SEARCH_QUERY = reuters_config['search_query']
                if 'target_categories' in reuters_config:
                    self.REUTERS_TARGET_CATEGORIES = reuters_config['target_categories']
                if 'exclude_keywords' in reuters_config:
                    self.REUTERS_EXCLUDE_KEYWORDS = reuters_config['exclude_keywords']
                if 'max_pages' in reuters_config:
                    self.REUTERS_MAX_PAGES = reuters_config['max_pages']
            
            # Google Docs設定
            if 'google_docs' in news_config:
                gdocs_config = news_config['google_docs']
                if 'enabled' in gdocs_config:
                    self.GOOGLE_DOCS_ENABLED = gdocs_config['enabled']
                if 'credentials_path' in gdocs_config:
                    self.GOOGLE_DOCS_CREDENTIALS_PATH = gdocs_config['credentials_path']
    
    def _validate_configuration(self):
        """設定値の検証"""
        # 基本設定の検証
        ConfigValidator.validate_positive_integer(self.NEWS_HOURS_LIMIT, "NEWS_HOURS_LIMIT")
        ConfigValidator.validate_positive_integer(self.MAX_NEWS_PAGES, "MAX_NEWS_PAGES")
        ConfigValidator.validate_positive_integer(self.REUTERS_MAX_PAGES, "REUTERS_MAX_PAGES")
        
        # URL設定の検証
        ConfigValidator.validate_url(self.REUTERS_BASE_URL, "REUTERS_BASE_URL")
        ConfigValidator.validate_url(self.REUTERS_SEARCH_URL, "REUTERS_SEARCH_URL")
        
        # 文字列設定の検証
        ConfigValidator.validate_string(self.REUTERS_SEARCH_QUERY, "REUTERS_SEARCH_QUERY")
        ConfigValidator.validate_string(self.GOOGLE_DOCS_CREDENTIALS_PATH, "GOOGLE_DOCS_CREDENTIALS_PATH")
        
        # リスト設定の検証
        ConfigValidator.validate_list(self.REUTERS_TARGET_CATEGORIES, "REUTERS_TARGET_CATEGORIES", min_items=1)
        ConfigValidator.validate_list(self.REUTERS_EXCLUDE_KEYWORDS, "REUTERS_EXCLUDE_KEYWORDS")
        
        # ファイルパス検証（Google Docs有効時のみ）
        if self.GOOGLE_DOCS_ENABLED:
            ConfigValidator.validate_file_path(
                self.GOOGLE_DOCS_CREDENTIALS_PATH,
                "GOOGLE_DOCS_CREDENTIALS_PATH",
                must_exist=True
            )


@dataclass
class WebScrapingConfig(BaseConfig):
    """Webスクレイピング設定"""
    
    # ユーザーエージェント設定
    USER_AGENT_STRING: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    
    # WebDriver設定
    WEBDRIVER_IMPLICIT_WAIT: int = 15
    WEBDRIVER_PAGE_LOAD_TIMEOUT: int = 120
    WEBDRIVER_HEADLESS: bool = True
    
    # タイミング設定
    SCRAPING_DELAY_SECONDS: int = 7
    PAGE_DELAY_SECONDS: int = 1
    HTTP_REQUEST_TIMEOUT: int = 15
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'web_scraping' in config_dict:
            ws_config = config_dict['web_scraping']
            
            if 'user_agent' in ws_config:
                self.USER_AGENT_STRING = ws_config['user_agent']
            if 'webdriver_implicit_wait' in ws_config:
                self.WEBDRIVER_IMPLICIT_WAIT = ws_config['webdriver_implicit_wait']
            if 'webdriver_page_load_timeout' in ws_config:
                self.WEBDRIVER_PAGE_LOAD_TIMEOUT = ws_config['webdriver_page_load_timeout']
            if 'webdriver_headless' in ws_config:
                self.WEBDRIVER_HEADLESS = ws_config['webdriver_headless']
            if 'scraping_delay_seconds' in ws_config:
                self.SCRAPING_DELAY_SECONDS = ws_config['scraping_delay_seconds']
            if 'page_delay_seconds' in ws_config:
                self.PAGE_DELAY_SECONDS = ws_config['page_delay_seconds']
            if 'http_request_timeout' in ws_config:
                self.HTTP_REQUEST_TIMEOUT = ws_config['http_request_timeout']
    
    def _validate_configuration(self):
        """設定値の検証"""
        ConfigValidator.validate_string(self.USER_AGENT_STRING, "USER_AGENT_STRING")
        ConfigValidator.validate_positive_integer(self.WEBDRIVER_IMPLICIT_WAIT, "WEBDRIVER_IMPLICIT_WAIT")
        ConfigValidator.validate_positive_integer(self.WEBDRIVER_PAGE_LOAD_TIMEOUT, "WEBDRIVER_PAGE_LOAD_TIMEOUT")
        ConfigValidator.validate_positive_integer(self.SCRAPING_DELAY_SECONDS, "SCRAPING_DELAY_SECONDS")
        ConfigValidator.validate_positive_integer(self.PAGE_DELAY_SECONDS, "PAGE_DELAY_SECONDS")
        ConfigValidator.validate_positive_integer(self.HTTP_REQUEST_TIMEOUT, "HTTP_REQUEST_TIMEOUT")