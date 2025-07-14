"""
APIクライアントの抽象化モジュール
"""

import abc
from typing import Dict, List, Optional, Any
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import logging
from bs4 import BeautifulSoup

from config import Config
from utils import retry_api_call, retry_network_operation, safe_request, DataValidator
from logger import get_metrics_logger, log_execution_time, log_api_call

class BaseAPIClient(abc.ABC):
    """API クライアントの基底クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_logger = get_metrics_logger()
    
    @abc.abstractmethod
    def fetch_data(self, *args, **kwargs) -> Any:
        """データ取得の抽象メソッド"""
        pass
    
    def _log_api_call(self, api_name: str, success: bool, **kwargs):
        """API呼び出しのログ記録"""
        self.metrics_logger.log_api_metrics(
            api_name=api_name,
            endpoint=kwargs.get('endpoint', 'unknown'),
            status_code=200 if success else 500,
            response_time=kwargs.get('response_time', 0),
            success=success
        )

class YFinanceClient(BaseAPIClient):
    """Yahoo Finance APIクライアント"""
    
    @log_execution_time("yfinance_fetch_ticker_data")
    @retry_api_call(max_attempts=3)
    def fetch_ticker_data(self, symbol: str, period: str = "1d", 
                         interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        ティッカーのデータを取得
        
        Args:
            symbol: ティッカーシンボル
            period: データ期間
            interval: データ間隔
        
        Returns:
            DataFrame: 取得したデータ
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if DataValidator.validate_chart_data(data, symbol):
                self._log_api_call("yfinance", True, endpoint=f"ticker/{symbol}")
                self.metrics_logger.log_data_metrics(
                    data_type="ticker_data",
                    record_count=len(data),
                    symbol=symbol,
                    period=period,
                    interval=interval
                )
                return data
            else:
                self.logger.warning(f"Invalid data received for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            self._log_api_call("yfinance", False, endpoint=f"ticker/{symbol}")
            raise
    
    @log_execution_time("yfinance_fetch_ticker_info")
    @retry_api_call(max_attempts=3)
    def fetch_ticker_info(self, symbol: str) -> Optional[Dict]:
        """ティッカーの情報を取得"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info and 'regularMarketPrice' in info:
                self._log_api_call("yfinance", True, endpoint=f"info/{symbol}")
                return info
            else:
                self.logger.warning(f"No valid info for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to fetch info for {symbol}: {e}")
            self._log_api_call("yfinance", False, endpoint=f"info/{symbol}")
            raise

class InvestpyClient(BaseAPIClient):
    """Investpy APIクライアント"""
    
    @log_execution_time("investpy_fetch_economic_calendar")
    @retry_api_call(max_attempts=3)
    def fetch_economic_calendar(self, countries: List[str], 
                               from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """
        経済カレンダーを取得
        
        Args:
            countries: 対象国のリスト
            from_date: 開始日 (YYYY-MM-DD)
            to_date: 終了日 (YYYY-MM-DD)
        
        Returns:
            DataFrame: 経済指標データ
        """
        try:
            import investpy
            
            data = investpy.economic_calendar(
                countries=countries,
                from_date=from_date,
                to_date=to_date
            )
            
            if data is not None and not data.empty:
                self._log_api_call("investpy", True, endpoint="economic_calendar")
                self.metrics_logger.log_data_metrics(
                    data_type="economic_calendar",
                    record_count=len(data),
                    countries=countries,
                    date_range=f"{from_date}_to_{to_date}"
                )
                return data
            else:
                self.logger.warning("No economic calendar data received")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to fetch economic calendar: {e}")
            self._log_api_call("investpy", False, endpoint="economic_calendar")
            raise

class GeminiClient(BaseAPIClient):
    """Google Gemini APIクライアント"""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Geminiモデルの初期化"""
        try:
            import google.generativeai as genai
            
            # 設定から優先モデルを取得
            for model_name in self.config.GEMINI_PREFERRED_MODELS:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.logger.info(f"Successfully initialized Gemini model: {model_name}")
                    return model
                except Exception as e:
                    self.logger.warning(f"Failed to initialize model {model_name}: {e}")
                    continue
            
            raise RuntimeError("No Gemini models could be initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    @log_execution_time("gemini_generate_content")
    @retry_api_call(max_attempts=2)
    def generate_content(self, prompt: str, max_length: Optional[int] = None) -> Optional[str]:
        """
        コンテンツ生成
        
        Args:
            prompt: プロンプト
            max_length: 最大文字数
        
        Returns:
            str: 生成されたコンテンツ
        """
        try:
            # テキスト長制限
            if max_length is None:
                max_length = self.config.AI_TEXT_LIMIT
            
            if len(prompt) > max_length:
                prompt = prompt[:max_length]
                self.logger.warning(f"Prompt truncated to {max_length} characters")
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                self._log_api_call("gemini", True, endpoint="generate_content")
                self.metrics_logger.log_data_metrics(
                    data_type="ai_content",
                    record_count=1,
                    prompt_length=len(prompt),
                    response_length=len(response.text)
                )
                return response.text
            else:
                self.logger.warning("No content generated by Gemini")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate content: {e}")
            self._log_api_call("gemini", False, endpoint="generate_content")
            raise

class ReutersClient(BaseAPIClient):
    """Reuters ニュースクライアント"""
    
    @log_execution_time("reuters_search_news")
    @retry_network_operation(max_attempts=3)
    def search_news(self, query: str, max_pages: int = 5) -> List[Dict]:
        """
        Reutersニュース検索
        
        Args:
            query: 検索クエリ
            max_pages: 最大ページ数
        
        Returns:
            List[Dict]: ニュース記事のリスト
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from bs4 import BeautifulSoup
            
            # WebDriverの設定
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--user-agent={self.config.USER_AGENT_STRING}")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(self.config.WEBDRIVER_IMPLICIT_WAIT)
            driver.set_page_load_timeout(self.config.WEBDRIVER_PAGE_LOAD_TIMEOUT)
            
            articles = []
            
            try:
                for page in range(1, max_pages + 1):
                    search_url = f"{self.config.REUTERS_SEARCH_URL}?blob={query}&sortBy=date&page={page}"
                    
                    self.logger.info(f"Fetching Reuters page {page}: {search_url}")
                    driver.get(search_url)
                    
                    # ページ読み込み待機
                    time.sleep(self.config.SCRAPING_DELAY_SECONDS)
                    
                    # BeautifulSoupでパース
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    page_articles = self._parse_reuters_articles(soup)
                    
                    if not page_articles:
                        self.logger.warning(f"No articles found on page {page}")
                        break
                    
                    articles.extend(page_articles)
                    time.sleep(self.config.PAGE_DELAY_SECONDS)
                
                # 重複除去
                unique_articles = self._remove_duplicate_articles(articles)
                
                self._log_api_call("reuters", True, endpoint="search")
                self.metrics_logger.log_data_metrics(
                    data_type="news_articles",
                    record_count=len(unique_articles),
                    pages_scraped=min(page, max_pages),
                    query=query
                )
                
                return unique_articles
                
            finally:
                driver.quit()
                
        except Exception as e:
            self.logger.error(f"Failed to search Reuters news: {e}")
            self._log_api_call("reuters", False, endpoint="search")
            raise
    
    def _parse_reuters_articles(self, soup: BeautifulSoup) -> List[Dict]:
        """Reutersページから記事を抽出"""
        articles = []
        
        # Reuters検索結果の記事要素を探す
        article_elements = soup.find_all(['div', 'article'], 
                                       class_=lambda x: x and any(keyword in x.lower() 
                                       for keyword in ['search-result', 'story', 'article']))
        
        for element in article_elements:
            try:
                # タイトルとリンクを取得
                title_link = element.find('a')
                if not title_link:
                    continue
                
                title = title_link.get_text(strip=True)
                relative_url = title_link.get('href')
                
                if not title or not relative_url:
                    continue
                
                # 完全なURLを構築
                if relative_url.startswith('/'):
                    url = self.config.REUTERS_BASE_URL + relative_url
                else:
                    url = relative_url
                
                # 日付を取得（可能な場合）
                date_element = element.find(['time', 'span'], 
                                          class_=lambda x: x and 'date' in x.lower())
                published_date = None
                if date_element:
                    published_date = date_element.get_text(strip=True)
                
                # カテゴリを推定
                category = self._estimate_category(title, element.get_text())
                
                # 除外キーワードチェック
                if self._should_exclude_article(title, category):
                    continue
                
                articles.append({
                    'title': title,
                    'url': url,
                    'published_date': published_date,
                    'category': category,
                    'country': 'US'  # Reutersの場合、主にUS
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to parse article element: {e}")
                continue
        
        return articles
    
    def _estimate_category(self, title: str, content: str) -> str:
        """記事のカテゴリを推定"""
        text = (title + " " + content).lower()
        
        for category in self.config.REUTERS_TARGET_CATEGORIES:
            if any(keyword in text for keyword in 
                   ['market', 'stock', 'economy', 'financial', 'business']):
                return category
        
        return "不明"
    
    def _should_exclude_article(self, title: str, category: str) -> bool:
        """記事を除外すべきかチェック"""
        text = title.lower()
        return any(keyword.lower() in text for keyword in self.config.REUTERS_EXCLUDE_KEYWORDS)
    
    def _remove_duplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """重複記事の除去"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles

class APIClientFactory:
    """APIクライアントのファクトリクラス"""
    
    @staticmethod
    def create_yfinance_client(config: Optional[Config] = None) -> YFinanceClient:
        """YFinanceクライアントを作成"""
        return YFinanceClient(config)
    
    @staticmethod
    def create_investpy_client(config: Optional[Config] = None) -> InvestpyClient:
        """Investpyクライアントを作成"""
        return InvestpyClient(config)
    
    @staticmethod
    def create_gemini_client(config: Optional[Config] = None) -> GeminiClient:
        """Geminiクライアントを作成"""
        return GeminiClient(config)
    
    @staticmethod
    def create_reuters_client(config: Optional[Config] = None) -> ReutersClient:
        """Reutersクライアントを作成"""
        return ReutersClient(config)