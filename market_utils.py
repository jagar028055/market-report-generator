"""
マーケットデータ処理のユーティリティ関数
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import re

from config import Config
from logger import get_metrics_logger, log_execution_time

class MarketDataProcessor:
    """マーケットデータ処理クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_logger = get_metrics_logger()
    
    @log_execution_time("process_ticker_data")
    def process_ticker_data(self, ticker: str, data: pd.DataFrame, 
                           asset_type: str) -> Optional[pd.DataFrame]:
        """
        ティッカーデータの処理
        
        Args:
            ticker: ティッカーシンボル
            data: 生データ
            asset_type: 資産タイプ
        
        Returns:
            pd.DataFrame: 処理済みデータ
        """
        if data is None or data.empty:
            self.logger.warning(f"No data for ticker {ticker}")
            return None
        
        try:
            # タイムゾーン変換
            processed_data = self._convert_timezone(data, asset_type)
            
            # データクリーニング
            processed_data = self._clean_ohlc_data(processed_data)
            
            # 技術指標の計算（必要に応じて）
            if len(processed_data) > 20:
                processed_data = self._add_technical_indicators(processed_data)
            
            self.metrics_logger.log_data_metrics(
                data_type="processed_ticker_data",
                record_count=len(processed_data),
                ticker=ticker,
                asset_type=asset_type
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Failed to process data for {ticker}: {e}")
            raise
    
    def _convert_timezone(self, data: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """タイムゾーン変換"""
        if asset_type == "US_STOCK":
            # 米国株は東京時間に変換
            if data.index.tz is None:
                data.index = data.index.tz_localize('America/New_York')
            
            tokyo_tz = pytz.timezone('Asia/Tokyo')
            data.index = data.index.tz_convert(tokyo_tz)
        
        elif asset_type == "24H_ASSET":
            # 24時間取引資産は日本時間に変換
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            
            tokyo_tz = pytz.timezone('Asia/Tokyo')
            data.index = data.index.tz_convert(tokyo_tz)
        
        return data
    
    def _clean_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """OHLCデータのクリーニング"""
        # 欠損値を除去
        data = data.dropna()
        
        # 異常値の検出と除去（価格が0以下の場合）
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # 重複インデックスの除去
        data = data[~data.index.duplicated(keep='last')]
        
        # インデックスでソート
        data = data.sort_index()
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """技術指標の追加"""
        try:
            # 短期移動平均（5日）
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            
            # ボリンジャーバンド
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = rolling_mean + (rolling_std * 2)
            data['BB_Lower'] = rolling_mean - (rolling_std * 2)
            
            # RSI（14日）
            data['RSI'] = self._calculate_rsi(data['Close'], 14)
            
        except Exception as e:
            self.logger.warning(f"Failed to add technical indicators: {e}")
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class WeekendHandler:
    """週末判定と営業日処理クラス"""
    
    @staticmethod
    def is_weekend_in_timezone(tz_name: str) -> bool:
        """指定タイムゾーンでの週末判定"""
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
        return now.weekday() >= 5  # 土曜日=5, 日曜日=6
    
    @staticmethod
    def is_us_market_open() -> bool:
        """米国市場の営業時間判定"""
        us_tz = pytz.timezone('America/New_York')
        now = us_tz.localize(datetime.now().replace(tzinfo=None))
        
        # 週末チェック
        if now.weekday() >= 5:
            return False
        
        # 営業時間チェック（9:30-16:00）
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    @staticmethod
    def get_last_business_day(timezone_name: str = 'America/New_York') -> datetime:
        """最終営業日を取得"""
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)
        
        # 現在が週末の場合、前の金曜日を返す
        if now.weekday() == 5:  # 土曜日
            return now - timedelta(days=1)
        elif now.weekday() == 6:  # 日曜日
            return now - timedelta(days=2)
        else:
            return now

class DataValidator:
    """データ検証クラス（utils.pyから移動・拡張）"""
    
    @staticmethod
    @log_execution_time("validate_market_data")
    def validate_market_data(data: Dict[str, Any]) -> bool:
        """マーケットデータの詳細検証"""
        if not data:
            logging.error("Market data is empty")
            return False
        
        required_fields = ['current', 'change', 'change_percent']
        
        for ticker, ticker_data in data.items():
            if not isinstance(ticker_data, dict):
                logging.error(f"Invalid data type for ticker {ticker}: {type(ticker_data)}")
                return False
            
            # 必須フィールドの確認
            missing_fields = [field for field in required_fields if field not in ticker_data]
            if missing_fields:
                logging.error(f"Missing fields for ticker {ticker}: {missing_fields}")
                return False
            
            # データ型と値の検証
            if not DataValidator._validate_numeric_fields(ticker, ticker_data):
                return False
        
        logging.debug(f"Market data validation passed for {len(data)} tickers")
        return True
    
    @staticmethod
    def _validate_numeric_fields(ticker: str, data: Dict[str, Any]) -> bool:
        """数値フィールドの検証"""
        numeric_fields = ['current', 'change', 'change_percent']
        
        for field in numeric_fields:
            if field in data and data[field] != 'N/A':
                try:
                    # パーセンテージとカンマを除去して数値変換
                    value_str = str(data[field]).replace('%', '').replace(',', '')
                    float(value_str)
                except (ValueError, TypeError):
                    logging.error(f"Invalid numeric value for {ticker}.{field}: {data[field]}")
                    return False
        
        return True
    
    @staticmethod
    def validate_news_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ニュース記事の検証とフィルタリング"""
        valid_articles = []
        
        for i, article in enumerate(articles):
            if DataValidator._is_valid_article(article, i):
                valid_articles.append(article)
        
        logging.info(f"Validated {len(valid_articles)} out of {len(articles)} articles")
        return valid_articles
    
    @staticmethod
    def _is_valid_article(article: Dict[str, Any], index: int) -> bool:
        """個別記事の検証"""
        required_fields = ['title', 'url']
        
        # 必須フィールドの確認
        for field in required_fields:
            if field not in article or not article[field]:
                logging.warning(f"Article {index} missing required field: {field}")
                return False
        
        # URLの形式チェック
        if not isinstance(article['url'], str) or not article['url'].startswith('http'):
            logging.warning(f"Article {index} has invalid URL: {article['url']}")
            return False
        
        # タイトルの長さチェック
        if len(article['title']) < 10:
            logging.warning(f"Article {index} has too short title: {article['title']}")
            return False
        
        return True

class EconomicDataProcessor:
    """経済指標データ処理クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.translations = self.config.INDICATOR_TRANSLATIONS
    
    @log_execution_time("process_economic_indicators")
    def process_economic_indicators(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        経済指標データの処理
        
        Args:
            data: 経済指標の生データ
        
        Returns:
            Dict: 処理済みの経済指標データ
        """
        if data is None or data.empty:
            return {"yesterday": [], "today_scheduled": []}
        
        try:
            # データの前処理
            data = self._preprocess_economic_data(data)
            
            # 昨日と今日のデータを分離
            yesterday_data = self._filter_yesterday_data(data)
            today_scheduled = self._filter_today_scheduled(data)
            
            return {
                "yesterday": yesterday_data,
                "today_scheduled": today_scheduled
            }
            
        except Exception as e:
            logging.error(f"Failed to process economic indicators: {e}")
            return {"yesterday": [], "today_scheduled": []}
    
    def _preprocess_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """経済指標データの前処理"""
        # 必要な列が存在するかチェック
        required_columns = ['date', 'event']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.warning(f"Missing columns in economic data: {missing_columns}")
            return pd.DataFrame()
        
        # 日付列の変換
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        # NaNを含む行を除去
        data = data.dropna(subset=['date', 'event'])
        
        return data
    
    def _filter_yesterday_data(self, data: pd.DataFrame) -> List[Dict]:
        """昨日のデータをフィルタリング"""
        yesterday = datetime.now().date() - timedelta(days=1)
        yesterday_data = data[data['date'].dt.date == yesterday]
        
        return self._convert_to_dict_list(yesterday_data)
    
    def _filter_today_scheduled(self, data: pd.DataFrame) -> List[Dict]:
        """今日予定のデータをフィルタリング"""
        today = datetime.now().date()
        today_data = data[data['date'].dt.date == today]
        
        return self._convert_to_dict_list(today_data)
    
    def _convert_to_dict_list(self, data: pd.DataFrame) -> List[Dict]:
        """DataFrameを辞書のリストに変換"""
        result = []
        
        for _, row in data.iterrows():
            indicator = {
                'time': row.get('time', 'N/A'),
                'name': self._translate_indicator_name(row.get('event', '')),
                'previous': row.get('previous', 'N/A'),
                'actual': row.get('actual', 'N/A'),
                'forecast': row.get('forecast', 'N/A')
            }
            result.append(indicator)
        
        return result
    
    def _translate_indicator_name(self, english_name: str) -> str:
        """経済指標名の翻訳"""
        return self.translations.get(english_name, english_name)

class TextProcessor:
    """テキスト処理ユーティリティ"""
    
    @staticmethod
    def clean_text_for_ai(text: str, max_length: int = 1800) -> str:
        """AI処理用のテキストクリーニング"""
        if not text:
            return ""
        
        # HTMLタグの除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # 多重スペースの正規化
        text = re.sub(r'\s+', ' ', text)
        
        # 改行の正規化
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 長さ制限
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """テキストからキーワードを抽出"""
        if not text:
            return []
        
        # 単語に分割
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 長さでフィルタリング
        keywords = [word for word in words if len(word) >= min_length]
        
        # 重複除去
        return list(set(keywords))

class ChartDataHelper:
    """チャートデータ処理ヘルパー"""
    
    @staticmethod
    def prepare_chart_metadata(ticker: str, data: pd.DataFrame, 
                             chart_type: str) -> Dict[str, Any]:
        """チャートメタデータの準備"""
        if data is None or data.empty:
            return {}
        
        return {
            'ticker': ticker,
            'chart_type': chart_type,
            'data_points': len(data),
            'date_range': {
                'start': data.index.min().isoformat() if not data.empty else None,
                'end': data.index.max().isoformat() if not data.empty else None
            },
            'price_range': {
                'min': float(data['Low'].min()) if 'Low' in data.columns else None,
                'max': float(data['High'].max()) if 'High' in data.columns else None
            }
        }
    
    @staticmethod
    def add_cache_buster(file_path: str) -> str:
        """キャッシュバスター付きのパス生成"""
        timestamp = int(datetime.now().timestamp())
        return f"{file_path}?v={timestamp}"