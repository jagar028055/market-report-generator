"""
経済指標データ取得専用クラス
"""

import investpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import os

from .base_fetcher import BaseDataFetcher
from ..utils.exceptions import EconomicDataError, NetworkError, ValidationError
from ..utils.error_handler import with_error_handling


class EconomicDataFetcher(BaseDataFetcher):
    """経済指標データ取得専用クラス"""
    
    def __init__(self, logger: Optional[Any] = None):
        super().__init__(logger)
        
        # 経済指標固有の設定
        self.target_countries = self.data_config.TARGET_CALENDAR_COUNTRIES
        self.indicator_translations = self.data_config.INDICATOR_TRANSLATIONS
        
        # 未訳の経済指標を保存するファイル
        self.untranslated_file = os.path.join(
            os.path.dirname(__file__), 
            "untranslated_indicators.txt"
        )
        
        self.logger.info(f"Initialized EconomicDataFetcher for countries: {self.target_countries}")
    
    def fetch_data(self, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """経済指標データを取得"""
        hours_limit = kwargs.get('hours_limit', 24)
        return self.get_economic_indicators(hours_limit)
    
    @with_error_handling()
    def get_economic_indicators(self, hours_limit: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """経済指標（過去と未来のデータ）を取得"""
        
        self.logger.info(f"Fetching economic indicators for past {hours_limit} hours and next {hours_limit} hours")
        
        try:
            # 経済カレンダーデータを取得・処理
            df_calendar = self._fetch_and_process_calendar(hours_limit)
            
            # 結果を分類
            economic_data = {"yesterday": [], "today_scheduled": []}
            
            if not df_calendar.empty:
                # 発表済みデータ
                announced_data = df_calendar[df_calendar['状態'] == '発表済み']
                for _, row in announced_data.iterrows():
                    economic_data["yesterday"].append({
                        "name": row['イベント'],
                        "time": row['日時(JST)'],
                        "previous": row.get('前回値', 'N/A'),
                        "actual": row.get('発表値', 'N/A'),
                        "forecast": row.get('予想値', 'N/A')
                    })
                
                # 発表予定データ
                scheduled_data = df_calendar[df_calendar['状態'] == '発表予定']
                for _, row in scheduled_data.iterrows():
                    economic_data["today_scheduled"].append({
                        "name": row['イベント'],
                        "time": row['日時(JST)'],
                        "previous": row.get('前回値', 'N/A'),
                        "forecast": row.get('予想値', 'N/A')
                    })
                
                self.logger.info(f"Economic indicators fetched: {len(economic_data['yesterday'])} announced, {len(economic_data['today_scheduled'])} scheduled")
            else:
                self.logger.warning("No economic calendar data available")
            
            return economic_data
            
        except Exception as e:
            raise EconomicDataError(f"Failed to fetch economic indicators: {e}")
    
    def _fetch_and_process_calendar(self, hours_limit: int) -> pd.DataFrame:
        """経済カレンダーを取得・処理"""
        
        # 1. 時間範囲を計算
        base_time_jst = datetime.now(self.jst)
        past_limit_jst = base_time_jst - timedelta(hours=hours_limit)
        future_limit_jst = base_time_jst + timedelta(hours=hours_limit)
        
        # 休日の場合は次の営業日まで延長
        while future_limit_jst.weekday() >= 5:  # 5=Sat, 6=Sun
            future_limit_jst += timedelta(days=1)
        
        from_date = past_limit_jst.strftime('%d/%m/%Y')
        to_date = future_limit_jst.strftime('%d/%m/%Y')
        
        self.logger.debug(f"Fetching economic calendar from {from_date} to {to_date}")
        
        try:
            # 2. investpyから経済カレンダーを取得
            df_raw = investpy.economic_calendar(
                from_date=from_date,
                to_date=to_date,
                countries=self.target_countries
            )
            
            if df_raw.empty:
                self.logger.warning("No economic calendar data found")
                return pd.DataFrame()
            
            # 3. データの処理
            df_processed = self._process_calendar_data(df_raw, base_time_jst, past_limit_jst, future_limit_jst)
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Failed to fetch economic calendar: {e}")
            return pd.DataFrame()
    
    def _process_calendar_data(
        self, 
        df_raw: pd.DataFrame, 
        base_time_jst: datetime, 
        past_limit_jst: datetime, 
        future_limit_jst: datetime
    ) -> pd.DataFrame:
        """経済カレンダーデータを処理"""
        
        df_processed = df_raw.copy()
        
        # 'time' が 'All Day' や空欄でない行のみを対象
        df_processed = df_processed[df_processed['time'].str.contains(':', na=False)].copy()
        
        if df_processed.empty:
            self.logger.warning("No time-specific economic events found")
            return pd.DataFrame()
        
        # 日付と時刻を結合してdatetimeオブジェクトに変換
        try:
            df_processed['datetime_utc'] = pd.to_datetime(
                df_processed['date'] + ' ' + df_processed['time'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            ).dt.tz_localize('Asia/Tokyo')  # investpyの時刻は東京時間とみなす
            
            # 変換失敗行を削除
            df_processed.dropna(subset=['datetime_utc'], inplace=True)
            
            if df_processed.empty:
                self.logger.warning("No valid datetime data after processing")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error processing datetime data: {e}")
            return pd.DataFrame()
        
        # 時間範囲でフィルタリング
        df_filtered = df_processed[
            (df_processed['datetime_utc'] >= past_limit_jst) &
            (df_processed['datetime_utc'] <= future_limit_jst)
        ].copy()
        
        if df_filtered.empty:
            self.logger.warning("No economic events in the specified time range")
            return pd.DataFrame()
        
        # 発表済み/発表予定のステータスを追加
        df_filtered['状態'] = np.where(
            df_filtered['datetime_utc'] < base_time_jst, 
            '発表済み', 
            '発表予定'
        )
        
        # 表示用のJST日時列を作成
        df_filtered['日時(JST)'] = df_filtered['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M')
        
        # 列名を日本語に変換
        column_rename_map = {
            'zone': '国',
            'event': 'イベント',
            'importance': '重要度',
            'actual': '発表値',
            'forecast': '予想値',
            'previous': '前回値'
        }
        df_filtered.rename(columns=column_rename_map, inplace=True)
        
        # 最終的な列を選択
        final_cols = ['状態', '日時(JST)', '国', '重要度', 'イベント', '発表値', '予想値', '前回値']
        df_final = df_filtered[[col for col in final_cols if col in df_filtered.columns]]
        
        # 指標名翻訳
        df_final = self._translate_indicators(df_final)
        
        return df_final.sort_values(by='日時(JST)')
    
    def _translate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """経済指標名を翻訳"""
        if 'イベント' not in df.columns:
            return df
        
        # 英語名を保持
        df['イベント_EN'] = df['イベント']
        
        # 翻訳を適用
        df['イベント'] = df['イベント_EN'].apply(
            lambda x: self.indicator_translations.get(x, x)
        )
        
        # 未訳の指標を記録
        untranslated = set(df[df['イベント'] == df['イベント_EN']]['イベント_EN'].unique())
        if untranslated:
            self._log_untranslated_indicators(untranslated)
        
        return df
    
    def _log_untranslated_indicators(self, indicators: Set[str]):
        """未訳の経済指標名を重複なくファイルに追記"""
        if not indicators:
            return
        
        try:
            # 既存の未訳指標を読み込み
            existing = set()
            if os.path.exists(self.untranslated_file):
                with open(self.untranslated_file, 'r', encoding='utf-8') as f:
                    existing = {line.strip() for line in f if line.strip()}
            
            # 新しい未訳指標のみを追記
            new_items = indicators - existing
            if new_items:
                # ディレクトリが存在しない場合は作成
                os.makedirs(os.path.dirname(self.untranslated_file), exist_ok=True)
                
                with open(self.untranslated_file, 'a', encoding='utf-8') as f:
                    for item in sorted(new_items):
                        f.write(item + "\n")
                
                self.logger.info(f"Logged {len(new_items)} new untranslated indicator(s)")
                
        except Exception as e:
            self.logger.warning(f"Unable to log untranslated indicators: {e}")
    
    @with_error_handling()
    def get_specific_indicator(self, indicator_name: str, days_back: int = 30) -> pd.DataFrame:
        """特定の経済指標の履歴データを取得"""
        
        self.logger.info(f"Fetching historical data for indicator: {indicator_name}")
        
        try:
            # 日付範囲を計算
            end_date = datetime.now(self.jst)
            start_date = end_date - timedelta(days=days_back)
            
            from_date = start_date.strftime('%d/%m/%Y')
            to_date = end_date.strftime('%d/%m/%Y')
            
            # 経済カレンダーから特定の指標を取得
            df_calendar = investpy.economic_calendar(
                from_date=from_date,
                to_date=to_date,
                countries=self.target_countries
            )
            
            if df_calendar.empty:
                return pd.DataFrame()
            
            # 指標名でフィルタリング（英語名と日本語名の両方で検索）
            english_name = None
            for eng, jpn in self.indicator_translations.items():
                if jpn == indicator_name:
                    english_name = eng
                    break
            
            if english_name:
                df_filtered = df_calendar[df_calendar['event'] == english_name]
            else:
                df_filtered = df_calendar[df_calendar['event'] == indicator_name]
            
            if df_filtered.empty:
                self.logger.warning(f"No data found for indicator: {indicator_name}")
                return pd.DataFrame()
            
            # 発表値があるデータのみを取得
            df_filtered = df_filtered[df_filtered['actual'].notna()]
            
            # 日付でソート
            df_filtered = df_filtered.sort_values('date')
            
            self.logger.info(f"Retrieved {len(df_filtered)} records for {indicator_name}")
            return df_filtered
            
        except Exception as e:
            raise EconomicDataError(f"Failed to fetch data for indicator {indicator_name}: {e}")
    
    def get_indicator_translation(self, english_name: str) -> str:
        """英語の指標名から日本語名を取得"""
        return self.indicator_translations.get(english_name, english_name)
    
    def add_indicator_translation(self, english_name: str, japanese_name: str):
        """指標翻訳を追加"""
        self.indicator_translations[english_name] = japanese_name
        self.logger.info(f"Added indicator translation: {english_name} -> {japanese_name}")
    
    def get_all_translations(self) -> Dict[str, str]:
        """すべての指標翻訳を取得"""
        return self.indicator_translations.copy()
    
    def get_untranslated_indicators(self) -> List[str]:
        """未訳の指標名リストを取得"""
        if not os.path.exists(self.untranslated_file):
            return []
        
        try:
            with open(self.untranslated_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Error reading untranslated indicators file: {e}")
            return []
    
    def clear_untranslated_indicators(self):
        """未訳指標ファイルをクリア"""
        try:
            if os.path.exists(self.untranslated_file):
                os.remove(self.untranslated_file)
                self.logger.info("Cleared untranslated indicators file")
        except Exception as e:
            self.logger.error(f"Error clearing untranslated indicators file: {e}")
    
    def get_indicator_summary(self, economic_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """経済指標の概要を取得"""
        
        all_indicators = economic_data.get('yesterday', []) + economic_data.get('today_scheduled', [])
        
        if not all_indicators:
            return {
                "total_indicators": 0,
                "announced_count": 0,
                "scheduled_count": 0,
                "countries": [],
                "most_recent": None
            }
        
        # 発表済み・予定の件数
        announced_count = len(economic_data.get('yesterday', []))
        scheduled_count = len(economic_data.get('today_scheduled', []))
        
        # 最新の発表
        most_recent = None
        if economic_data.get('yesterday'):
            most_recent = economic_data['yesterday'][-1]
        
        return {
            "total_indicators": len(all_indicators),
            "announced_count": announced_count,
            "scheduled_count": scheduled_count,
            "countries": self.target_countries,
            "most_recent": most_recent
        }
    
    def validate_economic_data(self, economic_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """経済指標データの妥当性をチェック"""
        
        if not isinstance(economic_data, dict):
            self.logger.error("Economic data must be a dictionary")
            return False
        
        required_keys = ['yesterday', 'today_scheduled']
        for key in required_keys:
            if key not in economic_data:
                self.logger.error(f"Missing required key: {key}")
                return False
            
            if not isinstance(economic_data[key], list):
                self.logger.error(f"Key '{key}' must be a list")
                return False
        
        # 各指標の必要フィールドをチェック
        required_fields = ['name', 'time']
        for category, indicators in economic_data.items():
            for i, indicator in enumerate(indicators):
                if not isinstance(indicator, dict):
                    self.logger.error(f"Indicator {i} in {category} must be a dictionary")
                    return False
                
                for field in required_fields:
                    if field not in indicator:
                        self.logger.error(f"Missing field '{field}' in indicator {i} of {category}")
                        return False
        
        return True


# ファクトリーに登録
from .base_fetcher import DataFetcherFactory
DataFetcherFactory.register_fetcher("economic", EconomicDataFetcher)