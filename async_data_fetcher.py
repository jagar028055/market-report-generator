"""
非同期データ取得モジュール
"""

import asyncio
import aiohttp
import concurrent.futures
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime

from config import Config
from logger import get_metrics_logger, log_execution_time
from api_clients import APIClientFactory, YFinanceClient, InvestpyClient, GeminiClient
from market_utils import MarketDataProcessor, WeekendHandler

@dataclass
class TaskResult:
    """タスク実行結果"""
    task_id: str
    success: bool
    data: Any
    execution_time: float
    error: Optional[str] = None

class AsyncDataFetcher:
    """非同期データ取得クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_logger = get_metrics_logger()
        self.semaphore = asyncio.Semaphore(self.config.MAX_WORKERS)
    
    @log_execution_time("fetch_all_market_data_async")
    async def fetch_all_market_data(self) -> Dict[str, Any]:
        """
        すべてのマーケットデータを非同期で取得
        
        Returns:
            Dict: 取得したすべてのデータ
        """
        start_time = time.time()
        
        # タスクを準備
        tasks = []
        
        # マーケットデータ取得タスク
        for ticker_name, symbol in self.config.MARKET_TICKERS.items():
            asset_type = self._get_asset_type(symbol)
            
            # イントラデイデータ
            tasks.append(self._create_ticker_task(
                f"intraday_{ticker_name}",
                symbol,
                period=f"{self.config.INTRADAY_PERIOD_DAYS}d",
                interval=self.config.INTRADAY_INTERVAL,
                asset_type=asset_type
            ))
            
            # 長期データ
            tasks.append(self._create_ticker_task(
                f"longterm_{ticker_name}",
                symbol,
                period=self.config.CHART_LONGTERM_PERIOD,
                interval="1d",
                asset_type=asset_type
            ))
        
        # セクターETFデータ取得タスク
        for etf_symbol in self.config.SECTOR_ETFS.keys():
            tasks.append(self._create_ticker_task(
                f"sector_{etf_symbol}",
                etf_symbol,
                period="5d",
                interval="1d",
                asset_type="US_STOCK"
            ))
        
        # 経済指標取得タスク
        tasks.append(self._create_economic_calendar_task())
        
        # すべてのタスクを並行実行
        results = await self._execute_tasks_with_semaphore(tasks)
        
        # 結果を整理
        organized_data = self._organize_results(results)
        
        total_time = time.time() - start_time
        self.logger.info(f"Async data fetching completed in {total_time:.2f} seconds")
        
        # メトリクス記録
        self.metrics_logger.log_function_metrics(
            func_name="fetch_all_market_data_async",
            execution_time=total_time,
            total_tasks=len(tasks),
            successful_tasks=sum(1 for r in results if r.success),
            failed_tasks=sum(1 for r in results if not r.success)
        )
        
        return organized_data
    
    async def _create_ticker_task(self, task_id: str, symbol: str, 
                                 period: str, interval: str, 
                                 asset_type: str) -> Coroutine:
        """ティッカーデータ取得タスクを作成"""
        return self._execute_with_semaphore(
            self._fetch_ticker_data_async,
            task_id, symbol, period, interval, asset_type
        )
    
    async def _create_economic_calendar_task(self) -> Coroutine:
        """経済カレンダー取得タスクを作成"""
        return self._execute_with_semaphore(
            self._fetch_economic_calendar_async,
            "economic_calendar"
        )
    
    async def _execute_with_semaphore(self, coro_func: Callable, *args) -> TaskResult:
        """セマフォを使用してタスクを実行"""
        async with self.semaphore:
            return await coro_func(*args)
    
    async def _fetch_ticker_data_async(self, task_id: str, symbol: str, 
                                      period: str, interval: str, 
                                      asset_type: str) -> TaskResult:
        """非同期ティッカーデータ取得"""
        start_time = time.time()
        
        try:
            # スレッドプールでyfinanceを実行（CPUバウンドなタスク）
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._fetch_ticker_sync, symbol, period, interval, asset_type
                )
                data = await loop.run_in_executor(None, future.result)
            
            execution_time = time.time() - start_time
            
            if data is not None:
                self.logger.debug(f"Successfully fetched {task_id}: {len(data)} records")
                return TaskResult(task_id, True, data, execution_time)
            else:
                return TaskResult(task_id, False, None, execution_time, "No data returned")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to fetch {task_id}: {e}")
            return TaskResult(task_id, False, None, execution_time, str(e))
    
    def _fetch_ticker_sync(self, symbol: str, period: str, 
                          interval: str, asset_type: str):
        """同期的なティッカーデータ取得（スレッドプール用）"""
        try:
            client = APIClientFactory.create_yfinance_client(self.config)
            data = client.fetch_ticker_data(symbol, period, interval)
            
            if data is not None:
                # データ処理
                processor = MarketDataProcessor(self.config)
                processed_data = processor.process_ticker_data(symbol, data, asset_type)
                return processed_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Sync fetch failed for {symbol}: {e}")
            raise
    
    async def _fetch_economic_calendar_async(self, task_id: str) -> TaskResult:
        """非同期経済カレンダー取得"""
        start_time = time.time()
        
        try:
            # スレッドプールでinvestpyを実行
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._fetch_economic_calendar_sync)
                data = await loop.run_in_executor(None, future.result)
            
            execution_time = time.time() - start_time
            
            if data is not None:
                return TaskResult(task_id, True, data, execution_time)
            else:
                return TaskResult(task_id, False, None, execution_time, "No economic data")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to fetch economic calendar: {e}")
            return TaskResult(task_id, False, None, execution_time, str(e))
    
    def _fetch_economic_calendar_sync(self):
        """同期的な経済カレンダー取得"""
        try:
            from datetime import datetime, timedelta
            
            # 日付範囲を設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            client = APIClientFactory.create_investpy_client(self.config)
            data = client.fetch_economic_calendar(
                countries=self.config.TARGET_CALENDAR_COUNTRIES,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Sync economic calendar fetch failed: {e}")
            raise
    
    async def _execute_tasks_with_semaphore(self, tasks: List[Coroutine]) -> List[TaskResult]:
        """セマフォ制御下でタスクを並行実行"""
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 例外をTaskResultに変換
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(TaskResult(
                        task_id=f"task_{i}",
                        success=False,
                        data=None,
                        execution_time=0,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Failed to execute tasks: {e}")
            return []
    
    def _get_asset_type(self, symbol: str) -> str:
        """シンボルから資産タイプを判定"""
        if symbol in self.config.ASSET_CLASSES["US_STOCK"]:
            return "US_STOCK"
        elif symbol in self.config.ASSET_CLASSES["24H_ASSET"]:
            return "24H_ASSET"
        else:
            return "US_STOCK"  # デフォルト
    
    def _organize_results(self, results: List[TaskResult]) -> Dict[str, Any]:
        """実行結果を整理してデータ構造に変換"""
        organized = {
            "market_data": {},
            "intraday_data": {},
            "longterm_data": {},
            "sector_data": {},
            "economic_indicators": None,
            "execution_summary": {
                "total_tasks": len(results),
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_execution_time": 0
            }
        }
        
        for result in results:
            if result.success:
                organized["execution_summary"]["successful_tasks"] += 1
                self._categorize_result(organized, result)
            else:
                organized["execution_summary"]["failed_tasks"] += 1
                self.logger.warning(f"Task {result.task_id} failed: {result.error}")
            
            organized["execution_summary"]["total_execution_time"] += result.execution_time
        
        return organized
    
    def _categorize_result(self, organized: Dict, result: TaskResult):
        """結果をカテゴリ別に分類"""
        task_id = result.task_id
        
        if task_id.startswith("intraday_"):
            ticker_name = task_id.replace("intraday_", "")
            organized["intraday_data"][ticker_name] = result.data
            
        elif task_id.startswith("longterm_"):
            ticker_name = task_id.replace("longterm_", "")
            organized["longterm_data"][ticker_name] = result.data
            
        elif task_id.startswith("sector_"):
            etf_symbol = task_id.replace("sector_", "")
            organized["sector_data"][etf_symbol] = result.data
            
        elif task_id == "economic_calendar":
            organized["economic_indicators"] = result.data

class AsyncNewsAggregator:
    """非同期ニュース収集クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_logger = get_metrics_logger()
    
    @log_execution_time("fetch_news_async")
    async def fetch_all_news(self) -> List[Dict[str, Any]]:
        """
        複数ソースから非同期でニュースを取得
        
        Returns:
            List[Dict]: ニュース記事のリスト
        """
        start_time = time.time()
        
        try:
            # Reutersニュース取得（現在の同期実装をそのまま使用）
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._fetch_reuters_news_sync)
                news_articles = await loop.run_in_executor(None, future.result)
            
            execution_time = time.time() - start_time
            
            self.metrics_logger.log_function_metrics(
                func_name="fetch_news_async",
                execution_time=execution_time,
                articles_count=len(news_articles) if news_articles else 0
            )
            
            return news_articles or []
            
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []
    
    def _fetch_reuters_news_sync(self) -> List[Dict[str, Any]]:
        """同期的なReutersニュース取得"""
        try:
            from api_clients import APIClientFactory
            
            client = APIClientFactory.create_reuters_client(self.config)
            articles = client.search_news(
                query=self.config.REUTERS_SEARCH_QUERY,
                max_pages=self.config.REUTERS_MAX_PAGES
            )
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Sync Reuters news fetch failed: {e}")
            raise

class AsyncCommentaryGenerator:
    """非同期コメント生成クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_logger = get_metrics_logger()
    
    @log_execution_time("generate_commentary_async")
    async def generate_commentary(self, market_data: Dict[str, Any], 
                                 news_articles: List[Dict[str, Any]]) -> str:
        """
        非同期でマーケットコメント生成
        
        Args:
            market_data: マーケットデータ
            news_articles: ニュース記事
        
        Returns:
            str: 生成されたコメント
        """
        try:
            # コメント生成をスレッドプールで実行
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._generate_commentary_sync, market_data, news_articles
                )
                commentary = await loop.run_in_executor(None, future.result)
            
            return commentary or "マーケットコメントの生成に失敗しました。"
            
        except Exception as e:
            self.logger.error(f"Failed to generate commentary: {e}")
            return "マーケットコメントの生成中にエラーが発生しました。"
    
    def _generate_commentary_sync(self, market_data: Dict[str, Any], 
                                 news_articles: List[Dict[str, Any]]) -> str:
        """同期的なコメント生成"""
        try:
            from market_utils import TextProcessor
            
            # プロンプト作成
            prompt = self._create_commentary_prompt(market_data, news_articles)
            
            # Geminiクライアントでコメント生成
            client = APIClientFactory.create_gemini_client(self.config)
            commentary = client.generate_content(prompt, self.config.AI_TEXT_LIMIT)
            
            return commentary
            
        except Exception as e:
            self.logger.error(f"Sync commentary generation failed: {e}")
            raise
    
    def _create_commentary_prompt(self, market_data: Dict[str, Any], 
                                 news_articles: List[Dict[str, Any]]) -> str:
        """コメント生成用プロンプトの作成"""
        # 基本的なプロンプト（既存のcommentary_generator.pyから移植）
        prompt = "以下のマーケットデータとニュースを基に、日本語でマーケット分析レポートを作成してください。\n\n"
        
        # マーケットデータ部分
        if market_data:
            prompt += "## マーケットデータ\n"
            for ticker, data in market_data.items():
                if isinstance(data, dict):
                    prompt += f"- {ticker}: {data.get('current', 'N/A')}\n"
        
        # ニュース部分
        if news_articles:
            prompt += "\n## 関連ニュース\n"
            for i, article in enumerate(news_articles[:5]):  # 最初の5件のみ
                prompt += f"- {article.get('title', '')}\n"
        
        prompt += "\n分析は簡潔で分かりやすく、投資判断に役立つ情報を含めてください。"
        
        return prompt

async def main_async_execution():
    """メイン非同期実行関数"""
    config = Config()
    
    # 非同期データ取得
    data_fetcher = AsyncDataFetcher(config)
    news_aggregator = AsyncNewsAggregator(config)
    commentary_generator = AsyncCommentaryGenerator(config)
    
    # 並行してデータとニュースを取得
    market_data_task = data_fetcher.fetch_all_market_data()
    news_task = news_aggregator.fetch_all_news()
    
    market_data, news_articles = await asyncio.gather(
        market_data_task, news_task, return_exceptions=True
    )
    
    # エラーハンドリング
    if isinstance(market_data, Exception):
        logging.error(f"Market data fetch failed: {market_data}")
        market_data = {}
    
    if isinstance(news_articles, Exception):
        logging.error(f"News fetch failed: {news_articles}")
        news_articles = []
    
    # コメント生成
    commentary = await commentary_generator.generate_commentary(
        market_data, news_articles
    )
    
    return {
        "market_data": market_data,
        "news_articles": news_articles,
        "commentary": commentary
    }

if __name__ == "__main__":
    # テスト実行用
    result = asyncio.run(main_async_execution())
    print(f"Async execution completed with {len(result['news_articles'])} news articles")