"""
データ取得の基底クラス
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pytz
import pandas as pd

from ..config import get_data_config, get_system_config
from ..utils.error_handler import ErrorHandler, RetryHandler
from ..utils.exceptions import DataFetchError, NetworkError, ValidationError


class BaseDataFetcher(ABC):
    """データ取得の基底クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_config = get_data_config()
        self.system_config = get_system_config()
        self.error_handler = ErrorHandler(self.logger)
        self.retry_handler = RetryHandler(self.error_handler)
        
        # タイムゾーン設定
        self.jst = pytz.timezone('Asia/Tokyo')
        self.ny_tz = pytz.timezone('America/New_York')
        self.utc = pytz.utc
        
        # 基本設定
        self.max_retries = 3
        self.timeout = 30
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> Any:
        """データを取得する（サブクラスで実装）"""
        pass
    
    def get_current_trading_day(self, timezone: str = "NY") -> datetime:
        """現在の取引日を取得"""
        if timezone == "NY":
            tz = self.ny_tz
        elif timezone == "JST":
            tz = self.jst
        else:
            tz = self.utc
        
        today = datetime.now(tz)
        
        # 週末の場合は前の営業日に調整
        if today.weekday() == 5:  # Saturday
            today = today - timedelta(days=1)
        elif today.weekday() == 6:  # Sunday
            today = today - timedelta(days=2)
        
        return today
    
    def get_previous_trading_day(self, base_date: datetime = None) -> datetime:
        """前の取引日を取得"""
        if base_date is None:
            base_date = self.get_current_trading_day()
        
        previous_date = base_date - timedelta(days=1)
        
        # 週末をスキップ
        while previous_date.weekday() >= 5:
            previous_date -= timedelta(days=1)
        
        return previous_date
    
    def validate_data(self, data: Any, data_type: str) -> bool:
        """データの妥当性をチェック"""
        if data is None:
            self.logger.error(f"{data_type} data is None")
            return False
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                self.logger.error(f"{data_type} DataFrame is empty")
                return False
            
            # 必要な列の存在チェック
            if data_type == "market_data":
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    self.logger.error(f"{data_type} missing required columns: {missing_cols}")
                    return False
        
        elif isinstance(data, dict):
            if not data:
                self.logger.error(f"{data_type} dictionary is empty")
                return False
        
        elif isinstance(data, list):
            if not data:
                self.logger.warning(f"{data_type} list is empty")
                return True  # 空のリストは有効とする場合もある
        
        return True
    
    def handle_fetch_error(self, error: Exception, context: str) -> None:
        """データ取得エラーの処理"""
        self.error_handler.handle_error(error, {'context': context})
        
        # エラーの種類に応じた処理
        if isinstance(error, NetworkError):
            self.logger.warning(f"Network error in {context}: {error}")
        elif isinstance(error, ValidationError):
            self.logger.error(f"Validation error in {context}: {error}")
        else:
            self.logger.error(f"Unexpected error in {context}: {error}")
    
    def log_fetch_result(self, result: Any, data_type: str, success: bool = True):
        """データ取得結果のログ"""
        if success:
            if isinstance(result, pd.DataFrame):
                self.logger.info(f"Successfully fetched {data_type}: {len(result)} rows")
            elif isinstance(result, dict):
                self.logger.info(f"Successfully fetched {data_type}: {len(result)} items")
            elif isinstance(result, list):
                self.logger.info(f"Successfully fetched {data_type}: {len(result)} items")
            else:
                self.logger.info(f"Successfully fetched {data_type}")
        else:
            self.logger.error(f"Failed to fetch {data_type}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """エラーの概要を取得"""
        return self.error_handler.get_error_summary()
    
    def clear_error_history(self):
        """エラー履歴をクリア"""
        self.error_handler.clear_history()
    
    def _convert_timezone(self, dt: datetime, target_tz: str) -> datetime:
        """タイムゾーンを変換"""
        if target_tz == "JST":
            return dt.astimezone(self.jst)
        elif target_tz == "NY":
            return dt.astimezone(self.ny_tz)
        elif target_tz == "UTC":
            return dt.astimezone(self.utc)
        else:
            raise ValueError(f"Unknown timezone: {target_tz}")
    
    def _normalize_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameの列を正規化"""
        if df.empty:
            return df
        
        # MultiIndexを平坦化
        if isinstance(df.columns, pd.MultiIndex):
            self.logger.debug("Flattening MultiIndex columns")
            df.columns = df.columns.get_level_values(0)
        
        return df
    
    def _clean_numeric_data(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """数値データのクリーニング"""
        if df.empty:
            return df
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NaNを持つ行を削除
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df
    
    def _validate_dataframe_structure(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """DataFrameの構造を検証"""
        if df.empty:
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        return True
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        return getattr(self.data_config, key, default)
    
    def set_retry_config(self, max_retries: int, timeout: int):
        """リトライ設定を更新"""
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger.info(f"Updated retry config: max_retries={max_retries}, timeout={timeout}")


class DataFetcherFactory:
    """データフェッチャーのファクトリークラス"""
    
    _fetchers = {}
    
    @classmethod
    def register_fetcher(cls, name: str, fetcher_class: type):
        """フェッチャーを登録"""
        cls._fetchers[name] = fetcher_class
    
    @classmethod
    def create_fetcher(cls, name: str, **kwargs) -> BaseDataFetcher:
        """フェッチャーを作成"""
        if name not in cls._fetchers:
            raise ValueError(f"Unknown fetcher: {name}")
        
        fetcher_class = cls._fetchers[name]
        return fetcher_class(**kwargs)
    
    @classmethod
    def get_available_fetchers(cls) -> List[str]:
        """利用可能なフェッチャーのリストを取得"""
        return list(cls._fetchers.keys())