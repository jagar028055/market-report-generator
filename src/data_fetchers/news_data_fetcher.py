"""
ニュースデータ取得専用クラス
"""

import time
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai

from .base_fetcher import BaseDataFetcher
from ..config import get_news_config, get_web_scraping_config, get_ai_config
from ..utils.exceptions import NewsDataError, WebScrapingError, NetworkError, APIError
from ..utils.error_handler import with_error_handling
from ..clients.google_docs_client import GoogleDocsClient


class NewsDataFetcher(BaseDataFetcher):
    """ニュースデータ取得専用クラス"""
    
    def __init__(self, logger: Optional[Any] = None):
        super().__init__(logger)
        
        # ニュース固有の設定
        self.news_config = get_news_config()
        self.web_config = get_web_scraping_config()
        self.ai_config = get_ai_config()
        
        # Chrome WebDriver設定
        self.chrome_options = self._setup_chrome_options()
        
        # Gemini AI設定
        self.gemini_model = self._setup_gemini_model()
        
        # Google Docsクライアント
        self.google_docs_client = None
        if self.news_config.GOOGLE_DOCS_ENABLED:
            try:
                self.google_docs_client = GoogleDocsClient(
                    credentials_path=self.news_config.GOOGLE_DOCS_CREDENTIALS_PATH
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google Docs client: {e}")
        
        self.logger.info("Initialized NewsDataFetcher")
    
    def _setup_chrome_options(self) -> Options:
        """Chrome WebDriverのオプション設定"""
        options = Options()
        
        if self.web_config.WEBDRIVER_HEADLESS:
            options.add_argument("--headless")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920x1080")
        options.add_argument(f'user-agent={self.web_config.USER_AGENT_STRING}')
        
        return options
    
    def _setup_gemini_model(self):
        """Gemini AIモデルの設定"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            self.logger.warning("GEMINI_API_KEY not set. News country classification will return 'OTHER'.")
            return None
        
        try:
            genai.configure(api_key=api_key)
            
            # 利用可能なモデルから選択
            chosen_model = None
            for model_name in self.ai_config.GEMINI_PREFERRED_MODELS:
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods and m.name == model_name:
                            chosen_model = genai.GenerativeModel(model_name)
                            break
                    if chosen_model:
                        break
                except:
                    continue
            
            if chosen_model is None:
                # フォールバック: 利用可能な最初のモデルを使用
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        chosen_model = genai.GenerativeModel(m.name)
                        break
            
            if chosen_model:
                self.logger.info(f"Gemini model initialized for news classification")
                return chosen_model
            else:
                self.logger.warning("No suitable Gemini model found")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini model: {e}")
            return None
    
    def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """ニュースデータを取得"""
        hours_limit = kwargs.get('hours_limit', self.news_config.NEWS_HOURS_LIMIT)
        max_pages = kwargs.get('max_pages', self.news_config.MAX_NEWS_PAGES)
        
        # Reuters記事を取得
        reuters_articles = self.scrape_reuters_news(
            query=self.news_config.REUTERS_SEARCH_QUERY,
            hours_limit=hours_limit,
            max_pages=max_pages,
            target_categories=self.news_config.REUTERS_TARGET_CATEGORIES,
            exclude_keywords=self.news_config.REUTERS_EXCLUDE_KEYWORDS
        )
        
        # Google Docs記事を取得（設定されている場合）
        google_docs_articles = []
        if self.google_docs_client and 'google_docs_id' in kwargs:
            google_docs_articles = self.get_google_docs_news(
                kwargs['google_docs_id'],
                hours_limit
            )
        
        # 記事を統合
        all_articles = reuters_articles + google_docs_articles
        
        self.log_fetch_result(all_articles, "news_articles")
        return all_articles
    
    @with_error_handling()
    def scrape_reuters_news(
        self,
        query: str,
        hours_limit: int = 24,
        max_pages: int = 5,
        items_per_page: int = 20,
        target_categories: List[str] = None,
        exclude_keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """ロイターのサイト内検索を利用して記事情報を収集"""
        
        articles_data = []
        processed_urls = set()
        
        if target_categories is None:
            target_categories = []
        if exclude_keywords is None:
            exclude_keywords = []
        
        driver = None
        self.logger.info("Starting Reuters news scraping")
        
        try:
            # WebDriverを初期化
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.implicitly_wait(self.web_config.WEBDRIVER_IMPLICIT_WAIT)
            driver.set_page_load_timeout(self.web_config.WEBDRIVER_PAGE_LOAD_TIMEOUT)
            
            # フィルター基準時刻
            time_threshold_jst = datetime.now(self.jst) - timedelta(hours=hours_limit)
            
            for page_num in range(max_pages):
                offset = page_num * items_per_page
                search_url = f"{self.news_config.REUTERS_SEARCH_URL}?query={requests.utils.quote(query)}&offset={offset}"
                
                self.logger.info(f"Scraping Reuters page {page_num + 1}/{max_pages}")
                
                try:
                    driver.get(search_url)
                    time.sleep(self.web_config.SCRAPING_DELAY_SECONDS)
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    articles_on_page = soup.find_all('li', attrs={"data-testid": "StoryCard"})
                    
                    if not articles_on_page:
                        if page_num == 0:
                            self.logger.warning("No articles found on first page. Site structure may have changed.")
                        break
                    
                    for article_li in articles_on_page:
                        article_data = self._extract_article_data(
                            article_li, 
                            time_threshold_jst,
                            target_categories,
                            exclude_keywords,
                            processed_urls
                        )
                        
                        if article_data:
                            articles_data.append(article_data)
                            processed_urls.add(article_data['url'])
                    
                    if len(articles_on_page) < items_per_page:
                        break
                    
                    time.sleep(self.web_config.PAGE_DELAY_SECONDS)
                    
                except Exception as e:
                    self.logger.error(f"Error scraping Reuters page {page_num + 1}: {e}")
                    continue
        
        except Exception as e:
            raise WebScrapingError(f"Reuters scraping failed: {e}")
        
        finally:
            if driver:
                driver.quit()
        
        self.logger.info(f"Reuters scraping completed: {len(articles_data)} articles")
        return articles_data
    
    def _extract_article_data(
        self,
        article_li,
        time_threshold_jst: datetime,
        target_categories: List[str],
        exclude_keywords: List[str],
        processed_urls: set
    ) -> Optional[Dict[str, Any]]:
        """記事データを抽出"""
        
        try:
            # タイトルとURLを抽出
            title, article_url = self._extract_title_and_url(article_li)
            if not title or not article_url:
                return None
            
            # 時刻を抽出
            article_time_jst = self._extract_article_time(article_li)
            if not article_time_jst:
                return None
            
            # カテゴリを抽出
            category_text = self._extract_category(article_li)
            
            # フィルター条件をチェック
            if not self._passes_filters(
                article_url, article_time_jst, title, category_text,
                processed_urls, time_threshold_jst, target_categories, exclude_keywords
            ):
                return None
            
            # 記事本文を取得
            body_text = self._scrape_article_body(article_url)
            
            # 国分類
            country_code = self.classify_country(f"{title}\n{body_text}")
            
            return {
                'title': title,
                'url': article_url,
                'published_jst': article_time_jst,
                'category': category_text,
                'country': country_code,
                'body': body_text if body_text else "[本文取得失敗/空]"
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting article data: {e}")
            return None
    
    def _extract_title_and_url(self, article_li) -> tuple:
        """タイトルとURLを抽出"""
        try:
            title_container = article_li.find('div', class_=re.compile(r'title__title'))
            link_element = title_container.find('a', attrs={"data-testid": "TitleLink"}) if title_container else None
            
            if not link_element:
                return None, None
            
            title = link_element.get_text(strip=True)
            article_url = link_element.get('href', '')
            
            if article_url.startswith('/'):
                article_url = self.news_config.REUTERS_BASE_URL + article_url
            
            return title, article_url
            
        except Exception as e:
            self.logger.error(f"Error extracting title and URL: {e}")
            return None, None
    
    def _extract_article_time(self, article_li) -> Optional[datetime]:
        """記事の時刻を抽出"""
        try:
            time_element = article_li.find('time', attrs={"data-testid": "DateLineText"})
            if time_element and time_element.has_attr('datetime'):
                dt_utc = datetime.fromisoformat(time_element.get('datetime').replace('Z', '+00:00'))
                return dt_utc.astimezone(self.jst)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting article time: {e}")
            return None
    
    def _extract_category(self, article_li) -> str:
        """カテゴリを抽出"""
        try:
            kicker = article_li.find('span', attrs={"data-testid": "KickerLabel"})
            if kicker:
                category_text_raw = kicker.get_text(strip=True)
                return category_text_raw.replace(" category", "").replace("Category", "").strip()
            
            return "不明"
            
        except Exception as e:
            self.logger.error(f"Error extracting category: {e}")
            return "不明"
    
    def _passes_filters(
        self,
        article_url: str,
        article_time_jst: datetime,
        title: str,
        category_text: str,
        processed_urls: set,
        time_threshold_jst: datetime,
        target_categories: List[str],
        exclude_keywords: List[str]
    ) -> bool:
        """フィルター条件をチェック"""
        
        # URL重複チェック
        if not article_url.startswith('http') or article_url in processed_urls:
            return False
        
        # 時刻チェック
        if article_time_jst < time_threshold_jst:
            return False
        
        # 除外キーワードチェック
        if any(keyword.lower() in title.lower() for keyword in exclude_keywords):
            return False
        
        # カテゴリチェック
        if target_categories and not any(
            tc.lower().replace("category", "").strip() in category_text.lower() 
            for tc in target_categories
        ):
            return False
        
        return True
    
    def _scrape_article_body(self, article_url: str) -> str:
        """記事本文を取得"""
        try:
            headers = {'User-Agent': self.web_config.USER_AGENT_STRING}
            response = requests.get(
                article_url, 
                headers=headers, 
                timeout=self.web_config.HTTP_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 記事本文コンテナを探す
            body_container = soup.find('div', class_='article-body__content__17Yit')
            if not body_container:
                body_container = soup.find('div', class_='article-body')
                if not body_container:
                    body_container = soup.find('div', class_='text__text__1FZnP')
            
            if not body_container:
                return ""
            
            # 段落を抽出
            paragraphs = [
                p_div.get_text(separator=' ', strip=True) 
                for p_div in body_container.find_all('div', attrs={"data-testid": lambda x: x and x.startswith('paragraph-')})
            ]
            
            if not paragraphs:
                paragraphs = [p.get_text(separator=' ', strip=True) for p in body_container.find_all('p')]
            
            article_text = '\n'.join(paragraphs)
            cleaned_text = re.sub(r'\s+', ' ', article_text).strip()
            
            return cleaned_text
            
        except Exception as e:
            self.logger.warning(f"Error scraping article body from {article_url}: {e}")
            return ""
    
    def classify_country(self, text: str) -> str:
        """Gemini APIを用いて記事の関連国を判定"""
        if not self.gemini_model:
            return "OTHER"
        
        prompt = (
            "以下のテキストは経済・マーケット関連ニュースのタイトルと本文です。"
            "主に関係する国を英語2文字(US, JP, CN, EU, UK など)で1つだけ回答してください。"
            "もし特定が難しければOTHERと答えてください。"
            "回答は国コードのみを1行で出力してください。\n\n---\n" + text[:self.ai_config.AI_TEXT_LIMIT] + "\n---\n"
        )
        
        try:
            response = self.gemini_model.generate_content(prompt)
            code = response.text.strip().upper()
            
            # 国コードの形式チェック
            if re.match(r"[A-Z]{2,3}", code):
                return code
            else:
                return "OTHER"
                
        except Exception as e:
            self.logger.warning(f"Gemini country classification error: {e}")
            return "OTHER"
    
    @with_error_handling()
    def get_google_docs_news(self, document_id: str, hours_limit: int = 24) -> List[Dict[str, Any]]:
        """Googleドキュメントからニュース記事を取得"""
        if not self.google_docs_client:
            self.logger.warning("Google Docs client not initialized")
            return []
        
        self.logger.info(f"Fetching news from Google Docs (document_id: {document_id})")
        
        try:
            articles = self.google_docs_client.fetch_news_articles(document_id, hours_limit)
            self.logger.info(f"Retrieved {len(articles)} articles from Google Docs")
            
            # 記事の詳細をログ出力
            for i, article in enumerate(articles, 1):
                self.logger.debug(f"  {i}. {article['title'][:50]}... ({article['published_jst'].strftime('%H:%M')})")
            
            return articles
            
        except Exception as e:
            raise NewsDataError(f"Failed to fetch news from Google Docs: {e}")
    
    def get_news_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ニュース記事の概要を取得"""
        if not articles:
            return {
                "total_articles": 0,
                "countries": {},
                "categories": {},
                "time_range": None
            }
        
        # 国別集計
        countries = {}
        for article in articles:
            country = article.get('country', 'OTHER')
            countries[country] = countries.get(country, 0) + 1
        
        # カテゴリ別集計
        categories = {}
        for article in articles:
            category = article.get('category', '不明')
            categories[category] = categories.get(category, 0) + 1
        
        # 時間範囲
        times = [article['published_jst'] for article in articles if 'published_jst' in article]
        time_range = None
        if times:
            time_range = {
                "earliest": min(times),
                "latest": max(times)
            }
        
        return {
            "total_articles": len(articles),
            "countries": countries,
            "categories": categories,
            "time_range": time_range
        }
    
    def filter_articles_by_country(self, articles: List[Dict[str, Any]], country: str) -> List[Dict[str, Any]]:
        """指定された国の記事をフィルター"""
        return [article for article in articles if article.get('country') == country]
    
    def filter_articles_by_category(self, articles: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
        """指定されたカテゴリの記事をフィルター"""
        return [article for article in articles if article.get('category') == category]


# ファクトリーに登録
from .base_fetcher import DataFetcherFactory
DataFetcherFactory.register_fetcher("news", NewsDataFetcher)