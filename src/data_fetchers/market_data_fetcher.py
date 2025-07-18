"""
市場データ取得専用クラス
"""

import yfinance as yf
import investpy
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np

from .base_fetcher import BaseDataFetcher
from ..utils.exceptions import MarketDataError, NetworkError, ValidationError
from ..utils.error_handler import with_error_handling


class MarketDataFetcher(BaseDataFetcher):
    """市場データ取得専用クラス"""
    
    def __init__(self, logger: Optional[Any] = None):
        super().__init__(logger)
        
        # 市場データ固有の設定
        self.tickers = self.data_config.MARKET_TICKERS
        self.sector_etfs = self.data_config.SECTOR_ETFS
        self.asset_classes = self.data_config.ASSET_CLASSES
        
        self.logger.info(f"Initialized MarketDataFetcher with {len(self.tickers)} tickers")
    
    def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """市場データを取得"""
        return self.get_market_data()
    
    @with_error_handling()
    def get_market_data(self) -> Dict[str, Any]:
        """主要指標の直近値、前日比、変化率を取得"""
        market_data = {}
        
        today = self.get_current_trading_day("NY").date()
        yesterday = self.get_previous_trading_day().date()
        
        self.logger.info(f"Fetching market data for {today} (previous: {yesterday})")
        
        for name, ticker in self.tickers.items():
            self.logger.debug(f"Fetching data for {name} ({ticker})")
            
            try:
                ticker_data = self._fetch_ticker_data(name, ticker, yesterday, today)
                market_data[name] = ticker_data
                
            except Exception as e:
                self.handle_fetch_error(e, f"market data for {name}")
                market_data[name] = {"current": "N/A", "change": "N/A", "change_percent": "N/A"}
        
        self.log_fetch_result(market_data, "market_data")
        return market_data
    
    def _fetch_ticker_data(self, name: str, ticker: str, yesterday: datetime, today: datetime) -> Dict[str, str]:
        """個別ティッカーのデータを取得"""
        # 米国2年金利の特別処理
        if name == "米国2年金利":
            return self._fetch_bond_data(name, ticker, yesterday, today)
        
        # 通常の株式データ取得
        return self._fetch_stock_data(name, ticker, yesterday, today)
    
    def _fetch_bond_data(self, name: str, ticker: str, yesterday: datetime, today: datetime) -> Dict[str, str]:
        """債券データを取得（investpy使用）"""
        try:
            data = investpy.get_bond_historical_data(
                bond='U.S. 2Y',
                from_date=yesterday.strftime('%d/%m/%Y'),
                to_date=today.strftime('%d/%m/%Y')
            )
            
            if not data.empty and len(data) >= 1:
                current_value = data['Close'].iloc[-1]
                
                if len(data) > 1:
                    previous_value = data['Close'].iloc[-2]
                else:
                    # 1日分のデータしかない場合、週末分を考慮
                    extended_data = investpy.get_bond_historical_data(
                        bond='U.S. 2Y',
                        from_date=(yesterday - timedelta(days=7)).strftime('%d/%m/%Y'),
                        to_date=today.strftime('%d/%m/%Y')
                    )
                    previous_value = extended_data['Close'].iloc[-2] if len(extended_data) > 1 else current_value
                
                change = current_value - previous_value
                change_percent = (change / previous_value) * 100 if previous_value != 0 else 0
                
                self.logger.debug(f"Bond data fetched for {name}: {current_value:.2f}%")
                
                return {
                    "current": f"{current_value:.2f}%",
                    "change": f"{change:+.2f}",
                    "change_percent": f"{change_percent:+.2f}%"
                }
            else:
                self.logger.warning(f"No investpy data available for {name}, falling back to yfinance")
                
        except Exception as e:
            self.logger.warning(f"investpy error for {name}: {e}, falling back to yfinance")
        
        # フォールバック: yfinanceを使用
        return self._fetch_stock_data(name, ticker, yesterday, today)
    
    def _fetch_stock_data(self, name: str, ticker: str, yesterday: datetime, today: datetime) -> Dict[str, str]:
        """株式データを取得（yfinance使用）"""
        try:
            data = yf.download(
                ticker,
                start=yesterday - timedelta(days=5),
                end=today + timedelta(days=1),
                progress=False
            )
            
            # データの正規化
            data = self._normalize_dataframe_columns(data)
            
            if not data.empty and len(data) >= 2:
                recent_data = data.tail(2)
                current_value = recent_data['Close'].iloc[-1]
                previous_value = recent_data['Close'].iloc[-2]
                change = current_value - previous_value
                change_percent = (change / previous_value) * 100 if previous_value != 0 else 0
                
                self.logger.debug(f"Stock data fetched for {name}: {current_value:.2f}")
                
                return {
                    "current": f"{current_value:.2f}",
                    "change": f"{change:.2f}",
                    "change_percent": f"{change_percent:.2f}%"
                }
            else:
                raise MarketDataError(f"Insufficient data for {name}")
                
        except Exception as e:
            raise MarketDataError(f"Failed to fetch stock data for {name}: {e}")
    
    @with_error_handling()
    def get_sector_etf_performance(self) -> Dict[str, Any]:
        """セクターETFの変化率を取得"""
        sector_performance = {}
        
        today = self.get_current_trading_day("NY").date()
        yesterday = self.get_previous_trading_day().date()
        
        self.logger.info(f"Fetching sector ETF performance for {today}")
        
        for ticker, name in self.sector_etfs.items():
            self.logger.debug(f"Fetching sector ETF data for {name} ({ticker})")
            
            try:
                data = yf.download(
                    ticker,
                    start=yesterday - timedelta(days=5),
                    end=today + timedelta(days=1),
                    progress=False,
                    auto_adjust=False
                )
                
                # データの正規化
                data = self._normalize_dataframe_columns(data)
                
                if not data.empty and len(data) >= 2:
                    recent_data = data.tail(2)
                    current_value = recent_data['Close'].iloc[-1]
                    previous_value = recent_data['Close'].iloc[-2]
                    change_percent = ((current_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
                    
                    sector_performance[name] = round(change_percent, 2) if change_percent is not None else None
                    self.logger.debug(f"Sector ETF data fetched for {name}: {change_percent:.2f}%")
                else:
                    sector_performance[name] = "N/A"
                    self.logger.warning(f"Insufficient data for sector ETF {name}")
                    
            except Exception as e:
                self.handle_fetch_error(e, f"sector ETF data for {name}")
                sector_performance[name] = "N/A"
        
        self.log_fetch_result(sector_performance, "sector_performance")
        return sector_performance
    
    @with_error_handling()
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """指定されたティッカーの過去データを取得"""
        self.logger.info(f"Fetching historical data for {ticker} ({period}, {interval})")
        
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            
            if data.empty:
                raise MarketDataError(f"No historical data available for {ticker}")
            
            # データの正規化とクリーニング
            data = self._normalize_dataframe_columns(data)
            data = self._clean_numeric_data(data, ['Open', 'High', 'Low', 'Close'])
            
            # 必要な列の存在確認
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not self._validate_dataframe_structure(data, required_cols):
                raise MarketDataError(f"Invalid data structure for {ticker}")
            
            # Volume列の処理
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)
            
            self.logger.info(f"Historical data fetched for {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            raise MarketDataError(f"Failed to fetch historical data for {ticker}: {e}")
    
    @with_error_handling()
    def get_intraday_data(self, ticker: str) -> pd.DataFrame:
        """指定されたティッカーのイントラデイデータを取得"""
        period_days = self.data_config.INTRADAY_PERIOD_DAYS
        interval = self.data_config.INTRADAY_INTERVAL
        
        self.logger.info(f"Fetching intraday data for {ticker} ({interval} for {period_days} days)")
        
        try:
            df_raw = yf.download(
                ticker,
                period=f"{period_days}d",
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            
            if df_raw.empty:
                raise MarketDataError(f"No intraday data available for {ticker}")
            
            # データの正規化とクリーニング
            df_cleaned = self._normalize_dataframe_columns(df_raw)
            df_cleaned = self._clean_numeric_data(df_cleaned, ['Open', 'High', 'Low', 'Close'])
            
            # 必要な列の存在確認
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not self._validate_dataframe_structure(df_cleaned, ohlcv_cols):
                raise MarketDataError(f"Invalid intraday data structure for {ticker}")
            
            # インデックスの処理
            df_processed = df_cleaned.reset_index()
            datetime_col = 'Datetime' if 'Datetime' in df_processed.columns else 'index'
            df_processed[datetime_col] = pd.to_datetime(df_processed[datetime_col])
            
            # タイムゾーン処理
            if df_processed[datetime_col].dt.tz is None:
                df_processed[datetime_col] = df_processed[datetime_col].dt.tz_localize(self.utc)
            else:
                df_processed[datetime_col] = df_processed[datetime_col].dt.tz_convert(self.utc)
            
            # 資産クラスに応じた処理
            df_final = self._process_intraday_by_asset_class(df_processed, ticker, datetime_col)
            
            if df_final.empty:
                raise MarketDataError(f"No data available for the target period for {ticker}")
            
            # 最終的な列の選択
            final_cols = ['日時', 'Open', 'High', 'Low', 'Close', 'Volume']
            final_cols_existing = [col for col in final_cols if col in df_final.columns]
            df_final = df_final[final_cols_existing]
            
            # インデックスを設定
            intraday_chart_data = df_final.set_index('日時')
            
            self.logger.info(f"Intraday data fetched for {ticker}: {len(intraday_chart_data)} rows")
            return intraday_chart_data
            
        except Exception as e:
            raise MarketDataError(f"Failed to fetch intraday data for {ticker}: {e}")
    
    def _process_intraday_by_asset_class(self, df: pd.DataFrame, ticker: str, datetime_col: str) -> pd.DataFrame:
        """資産クラスに応じたイントラデイデータの処理"""
        if self.data_config.is_us_stock(ticker):
            return self._process_us_stock_intraday(df, ticker, datetime_col)
        elif self.data_config.is_24h_asset(ticker):
            return self._process_24h_asset_intraday(df, ticker, datetime_col)
        else:
            # その他の資産：JSTに変換
            df['日時'] = df[datetime_col].dt.tz_convert(self.jst)
            return df
    
    def _process_us_stock_intraday(self, df: pd.DataFrame, ticker: str, datetime_col: str) -> pd.DataFrame:
        """米国株式のイントラデイデータ処理"""
        self.logger.debug(f"Processing US stock intraday data for {ticker}")
        
        df['日時_NY'] = df[datetime_col].dt.tz_convert(self.ny_tz)
        df['取引日_NY'] = df['日時_NY'].dt.normalize()
        
        # 最新の取引日のデータを取得
        latest_trading_day_ny = df['取引日_NY'].max()
        df_filtered = df[df['取引日_NY'] == latest_trading_day_ny].copy()
        
        # JSTに変換
        df_filtered['日時'] = df_filtered[datetime_col].dt.tz_convert(self.jst)
        
        self.logger.debug(f"Latest trading day (NY time) for {ticker}: {latest_trading_day_ny.strftime('%Y-%m-%d')}")
        
        return df_filtered
    
    def _process_24h_asset_intraday(self, df: pd.DataFrame, ticker: str, datetime_col: str) -> pd.DataFrame:
        """24時間取引資産のイントラデイデータ処理"""
        self.logger.debug(f"Processing 24H asset intraday data for {ticker}")
        
        df['日時_JST'] = df[datetime_col].dt.tz_convert(self.jst)
        
        # JST 7時開始の24時間データを取得
        now_jst = datetime.now(self.jst)
        today_7am_jst = now_jst.replace(hour=7, minute=0, second=0, microsecond=0)
        
        start_time_jst = today_7am_jst - timedelta(days=1) if now_jst < today_7am_jst else today_7am_jst
        end_time_jst = start_time_jst + timedelta(days=1)
        
        df_filtered = df[
            (df['日時_JST'] >= start_time_jst) & 
            (df['日時_JST'] < end_time_jst)
        ].copy()
        
        df_filtered['日時'] = df_filtered[datetime_col].dt.tz_convert(self.jst)
        
        self.logger.debug(f"24H asset extraction period for {ticker}: {start_time_jst.strftime('%Y-%m-%d %H:%M')} to {end_time_jst.strftime('%Y-%m-%d %H:%M')}")
        
        return df_filtered
    
    def get_ticker_symbol(self, name: str) -> str:
        """ティッカー名からシンボルを取得"""
        return self.tickers.get(name, name)
    
    def get_all_tickers(self) -> Dict[str, str]:
        """すべてのティッカーを取得"""
        return self.tickers.copy()
    
    def add_ticker(self, name: str, symbol: str):
        """ティッカーを追加"""
        self.tickers[name] = symbol
        self.logger.info(f"Added ticker: {name} = {symbol}")
    
    def remove_ticker(self, name: str):
        """ティッカーを削除"""
        if name in self.tickers:
            del self.tickers[name]
            self.logger.info(f"Removed ticker: {name}")
    
    def validate_ticker(self, ticker: str) -> bool:
        """ティッカーの有効性を確認"""
        try:
            # 簡単な検証：1日分のデータを取得してみる
            data = yf.download(ticker, period="1d", progress=False)
            return not data.empty
        except Exception as e:
            self.logger.warning(f"Ticker validation failed for {ticker}: {e}")
            return False

    def cleanup(self):
        """リソースをクリーンアップ"""
        self.logger.info("Cleaning up MarketDataFetcher resources.")
        # 現時点ではクリーンアップするリソースはないが、将来のためにメソッドを定義
        pass


# ファクトリーに登録
from .base_fetcher import DataFetcherFactory
DataFetcherFactory.register_fetcher("market", MarketDataFetcher)
