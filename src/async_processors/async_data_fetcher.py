"""
非同期データ取得クラス
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

from ..data_fetchers import MarketDataFetcher, NewsDataFetcher, EconomicDataFetcher
from ..config import get_system_config
from ..utils.exceptions import DataFetchError, NetworkError
from ..utils.error_handler import ErrorHandler


class AsyncDataFetcher:
    """非同期データ取得クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.system_config = get_system_config()
        self.error_handler = ErrorHandler(self.logger)
        
        # 各フェッチャーを初期化
        self.market_fetcher = MarketDataFetcher(self.logger)
        self.news_fetcher = NewsDataFetcher(self.logger)
        self.economic_fetcher = EconomicDataFetcher(self.logger)
        
        # 並行処理設定
        self.max_workers = self.system_config.MAX_WORKERS
        self.max_concurrent_requests = self.system_config.MAX_CONCURRENT_REQUESTS
        
        # セマフォを作成してリクエスト数を制限
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        self.logger.info(f"Initialized AsyncDataFetcher with {self.max_workers} workers")
    
    async def fetch_all_data(self, **kwargs) -> Dict[str, Any]:
        """すべてのデータを非同期で取得"""
        
        start_time = time.time()
        self.logger.info("Starting async data fetch for all sources")
        
        # 並行実行するタスクを作成
        tasks = [
            self._fetch_market_data_async(),
            self._fetch_economic_data_async(),
            self._fetch_sector_data_async(),
            self._fetch_news_data_async(**kwargs)
        ]
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を整理
        data = {}
        task_names = ["market_data", "economic_indicators", "sector_performance", "news_articles"]
        
        for i, (task_name, result) in enumerate(zip(task_names, results)):
            if isinstance(result, Exception):
                self.logger.error(f"Task {task_name} failed: {result}")
                self.error_handler.handle_error(result, {'task': task_name})
                data[task_name] = self._get_fallback_data(task_name)
            else:
                data[task_name] = result
                self.logger.info(f"Task {task_name} completed successfully")
        
        execution_time = time.time() - start_time
        self.logger.info(f"Async data fetch completed in {execution_time:.2f} seconds")
        
        return data
    
    async def _fetch_market_data_async(self) -> Dict[str, Any]:
        """市場データを非同期で取得"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            # ThreadPoolExecutorを使って同期関数を非同期実行
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.market_fetcher.get_market_data
                )
                return await future
    
    async def _fetch_economic_data_async(self) -> Dict[str, Any]:
        """経済指標データを非同期で取得"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.economic_fetcher.get_economic_indicators
                )
                return await future
    
    async def _fetch_sector_data_async(self) -> Dict[str, Any]:
        """セクターデータを非同期で取得"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.market_fetcher.get_sector_etf_performance
                )
                return await future
    
    async def _fetch_news_data_async(self, **kwargs) -> List[Dict[str, Any]]:
        """ニュースデータを非同期で取得"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.news_fetcher.fetch_data,
                    **kwargs
                )
                return await future
    
    async def fetch_chart_data_async(self, tickers: List[str] = None) -> Dict[str, Any]:
        """チャートデータを非同期で取得"""
        
        if tickers is None:
            tickers = list(self.market_fetcher.tickers.keys())
        
        # 2年金利を除外
        tickers = [ticker for ticker in tickers if ticker != "米国2年金利"]
        
        start_time = time.time()
        self.logger.info(f"Starting async chart data fetch for {len(tickers)} tickers")
        
        # 各ティッカーのタスクを作成
        tasks = []
        for ticker_name in tickers:
            ticker_symbol = self.market_fetcher.tickers.get(ticker_name)
            if ticker_symbol:
                task = self._fetch_ticker_chart_data_async(ticker_name, ticker_symbol)
                tasks.append(task)
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を整理
        chart_data = {}
        for i, (ticker_name, result) in enumerate(zip(tickers, results)):
            if isinstance(result, Exception):
                self.logger.error(f"Chart data fetch failed for {ticker_name}: {result}")
                self.error_handler.handle_error(result, {'ticker': ticker_name})
            else:
                chart_data[ticker_name] = result
                self.logger.info(f"Chart data fetched for {ticker_name}")
        
        execution_time = time.time() - start_time
        self.logger.info(f"Async chart data fetch completed in {execution_time:.2f} seconds")
        
        return chart_data
    
    async def _fetch_ticker_chart_data_async(self, ticker_name: str, ticker_symbol: str) -> Dict[str, Any]:
        """個別ティッカーのチャートデータを非同期で取得"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            # イントラデイとロングタームデータを並行取得
            with ThreadPoolExecutor(max_workers=2) as executor:
                intraday_future = loop.run_in_executor(
                    executor, 
                    self.market_fetcher.get_intraday_data,
                    ticker_symbol
                )
                
                longterm_future = loop.run_in_executor(
                    executor, 
                    self.market_fetcher.get_historical_data,
                    ticker_symbol,
                    "1y"
                )
                
                # 両方の結果を待つ
                intraday_data, longterm_data = await asyncio.gather(
                    intraday_future, 
                    longterm_future, 
                    return_exceptions=True
                )
                
                # 結果を整理
                result = {}
                
                if isinstance(intraday_data, Exception):
                    self.logger.warning(f"Intraday data fetch failed for {ticker_name}: {intraday_data}")
                    result["intraday"] = None
                else:
                    result["intraday"] = intraday_data
                
                if isinstance(longterm_data, Exception):
                    self.logger.warning(f"Long-term data fetch failed for {ticker_name}: {longterm_data}")
                    result["longterm"] = None
                else:
                    result["longterm"] = longterm_data
                
                return result
    
    async def fetch_multiple_news_sources_async(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数のニュースソースから非同期で記事を取得"""
        
        start_time = time.time()
        self.logger.info(f"Starting async news fetch from {len(sources)} sources")
        
        # 各ソースのタスクを作成
        tasks = []
        for source in sources:
            if source.get('type') == 'reuters':
                task = self._fetch_reuters_news_async(source)
            elif source.get('type') == 'google_docs':
                task = self._fetch_google_docs_news_async(source)
            else:
                self.logger.warning(f"Unknown news source type: {source.get('type')}")
                continue
            
            tasks.append(task)
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"News fetch failed for source {i}: {result}")
                self.error_handler.handle_error(result, {'source_index': i})
            else:
                all_articles.extend(result)
        
        execution_time = time.time() - start_time
        self.logger.info(f"Async news fetch completed in {execution_time:.2f} seconds, {len(all_articles)} articles total")
        
        return all_articles
    
    async def _fetch_reuters_news_async(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reuters記事を非同期で取得"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.news_fetcher.scrape_reuters_news,
                    source.get('query', ''),
                    source.get('hours_limit', 24),
                    source.get('max_pages', 3)
                )
                return await future
    
    async def _fetch_google_docs_news_async(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Google Docs記事を非同期で取得"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor, 
                    self.news_fetcher.get_google_docs_news,
                    source.get('document_id', ''),
                    source.get('hours_limit', 24)
                )
                return await future
    
    def _get_fallback_data(self, data_type: str) -> Any:
        """フォールバックデータを取得"""
        
        fallback_data = {
            "market_data": {},
            "economic_indicators": {"yesterday": [], "today_scheduled": []},
            "sector_performance": {},
            "news_articles": []
        }
        
        return fallback_data.get(data_type, {})
    
    async def fetch_with_timeout(self, coro: Coroutine, timeout: float = 30.0) -> Any:
        """タイムアウト付きでコルーチンを実行"""
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Operation timed out after {timeout} seconds")
            raise NetworkError(f"Operation timed out after {timeout} seconds")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクスを取得"""
        
        return {
            "max_workers": self.max_workers,
            "max_concurrent_requests": self.max_concurrent_requests,
            "error_summary": self.error_handler.get_error_summary()
        }
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        
        # エラー履歴をクリア
        self.error_handler.clear_history()
        
        # 各フェッチャーのクリーンアップ
        if hasattr(self.market_fetcher, 'cleanup'):
            self.market_fetcher.cleanup()
        if hasattr(self.news_fetcher, 'cleanup'):
            self.news_fetcher.cleanup()
        if hasattr(self.economic_fetcher, 'cleanup'):
            self.economic_fetcher.cleanup()
        
        self.logger.info("AsyncDataFetcher cleanup completed")


# 使用例関数
async def fetch_all_market_data():
    """すべての市場データを非同期で取得する使用例"""
    
    async_fetcher = AsyncDataFetcher()
    
    try:
        # すべてのデータを取得
        data = await async_fetcher.fetch_all_data()
        
        # チャートデータも取得
        chart_data = await async_fetcher.fetch_chart_data_async()
        
        # 結果を統合
        data['chart_data'] = chart_data
        
        return data
        
    except Exception as e:
        logging.error(f"Error in fetch_all_market_data: {e}")
        raise
    
    finally:
        async_fetcher.cleanup()


async def fetch_specific_tickers(tickers: List[str]):
    """特定のティッカーのデータを非同期で取得"""
    
    async_fetcher = AsyncDataFetcher()
    
    try:
        chart_data = await async_fetcher.fetch_chart_data_async(tickers)
        return chart_data
        
    except Exception as e:
        logging.error(f"Error in fetch_specific_tickers: {e}")
        raise
    
    finally:
        async_fetcher.cleanup()


# 同期インターフェース（後方互換性のため）
def fetch_all_data_sync(**kwargs) -> Dict[str, Any]:
    """同期インターフェースですべてのデータを取得"""
    
    async def _fetch():
        async_fetcher = AsyncDataFetcher()
        try:
            return await async_fetcher.fetch_all_data(**kwargs)
        finally:
            async_fetcher.cleanup()
    
    return asyncio.run(_fetch())


def fetch_chart_data_sync(tickers: List[str] = None) -> Dict[str, Any]:
    """同期インターフェースでチャートデータを取得"""
    
    async def _fetch():
        async_fetcher = AsyncDataFetcher()
        try:
            return await async_fetcher.fetch_chart_data_async(tickers)
        finally:
            async_fetcher.cleanup()
    
    return asyncio.run(_fetch())