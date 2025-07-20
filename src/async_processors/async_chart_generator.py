"""
非同期チャート生成クラス
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from ..chart_generators import CandlestickChartGenerator, SectorChartGenerator
from ..config import get_system_config
from ..utils.exceptions import ChartGenerationError
from ..utils.error_handler import ErrorHandler


class AsyncChartGenerator:
    """非同期チャート生成クラス"""
    
    def __init__(self, charts_dir: str = "charts", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.charts_dir = Path(charts_dir)
        self.system_config = get_system_config()
        self.error_handler = ErrorHandler(self.logger)
        
        # 各チャートジェネレーターを初期化
        self.candlestick_generator = CandlestickChartGenerator(str(self.charts_dir), self.logger)
        self.sector_generator = SectorChartGenerator(str(self.charts_dir), self.logger)
        
        # 並行処理設定
        self.max_workers = self.system_config.MAX_WORKERS
        self.max_concurrent_charts = self.system_config.MAX_CONCURRENT_REQUESTS
        
        # セマフォを作成してチャート生成数を制限
        self.semaphore = asyncio.Semaphore(self.max_concurrent_charts)
        
        self.logger.info(f"Initialized AsyncChartGenerator with {self.max_workers} workers")
    
    async def generate_all_charts(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """すべてのチャートを非同期で生成"""
        
        start_time = time.time()
        self.logger.info("Starting async chart generation for all data")
        
        # チャート生成タスクを作成
        tasks = []
        
        # 個別チャートタスク
        for ticker_name, data_set in chart_data.items():
            if isinstance(data_set, dict):
                # イントラデイチャート（HTML版とPNG版の両方を生成）
                if "intraday" in data_set and data_set["intraday"] is not None:
                    # HTMLチャート
                    task_html = self._generate_intraday_chart_async(
                        ticker_name, data_set["intraday"], 'interactive'
                    )
                    tasks.append(task_html)
                    
                    # PNGチャート
                    task_png = self._generate_intraday_chart_async(
                        ticker_name, data_set["intraday"], 'static'
                    )
                    tasks.append(task_png)
                
                # 長期チャート（HTML版とPNG版の両方を生成）
                if "longterm" in data_set and data_set["longterm"] is not None:
                    # HTMLチャート
                    task_html = self._generate_longterm_chart_async(
                        ticker_name, data_set["longterm"], 'interactive'
                    )
                    tasks.append(task_html)
                    
                    # PNGチャート
                    task_png = self._generate_longterm_chart_async(
                        ticker_name, data_set["longterm"], 'static'
                    )
                    tasks.append(task_png)
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を整理
        generated_charts = {"Intraday": [], "Long-Term": [], "Static-PNG": {}}
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Chart generation failed: {result}")
                self.error_handler.handle_error(result, {'operation': 'chart_generation'})
            elif result:
                chart_info = result
                if chart_info.get('type') == 'intraday':
                    if chart_info.get('interactive'):
                        generated_charts["Intraday"].append(chart_info)
                    else:
                        # PNGチャートの情報を保存
                        if chart_info['name'] not in generated_charts["Static-PNG"]:
                            generated_charts["Static-PNG"][chart_info['name']] = {}
                        generated_charts["Static-PNG"][chart_info['name']]['intraday'] = chart_info['path']
                elif chart_info.get('type') == 'longterm':
                    if chart_info.get('interactive'):
                        generated_charts["Long-Term"].append(chart_info)
                    else:
                        # PNGチャートの情報を保存
                        if chart_info['name'] not in generated_charts["Static-PNG"]:
                            generated_charts["Static-PNG"][chart_info['name']] = {}
                        generated_charts["Static-PNG"][chart_info['name']]['longterm'] = chart_info['path']
        
        execution_time = time.time() - start_time
        self.logger.info(f"Async chart generation completed in {execution_time:.2f} seconds")
        
        return generated_charts
    
    async def _generate_intraday_chart_async(
        self, 
        ticker_name: str, 
        data: Any,
        chart_type: str = 'interactive'
    ) -> Optional[Dict[str, Any]]:
        """イントラデイチャートを非同期で生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # チャートタイプに応じてファイル拡張子を決定
                extension = '.png' if chart_type == 'static' else '.html'
                filename = f"{ticker_name.replace(' ', '_')}_intraday{extension}"
                
                future = loop.run_in_executor(
                    executor,
                    self.candlestick_generator.generate_intraday_chart,
                    data,
                    ticker_name,
                    filename,
                    chart_type
                )
                
                chart_path = await future
                
                if chart_path:
                    sanitized_name = ticker_name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    return {
                        "id": f"{sanitized_name}-intraday",
                        "name": ticker_name,
                        "path": f"charts/{filename}",
                        "type": "intraday",
                        "interactive": chart_type == 'interactive'
                    }
                
                return None
    
    async def _generate_longterm_chart_async(
        self, 
        ticker_name: str, 
        data: Any,
        chart_type: str = 'interactive'
    ) -> Optional[Dict[str, Any]]:
        """長期チャートを非同期で生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # チャートタイプに応じてファイル拡張子を決定
                extension = '.png' if chart_type == 'static' else '.html'
                filename = f"{ticker_name.replace(' ', '_')}_longterm{extension}"
                
                future = loop.run_in_executor(
                    executor,
                    self.candlestick_generator.generate_longterm_chart,
                    data,
                    ticker_name,
                    filename,
                    chart_type
                )
                
                chart_path = await future
                
                if chart_path:
                    sanitized_name = ticker_name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    return {
                        "id": f"{sanitized_name}-longterm",
                        "name": ticker_name,
                        "path": f"charts/{filename}",
                        "type": "longterm",
                        "interactive": chart_type == 'interactive'
                    }
                
                return None
    
    async def generate_sector_chart_async(
        self, 
        sector_data: Dict[str, Any], 
        filename: str = "sector_performance_chart.html"
    ) -> Optional[str]:
        """セクターチャートを非同期で生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # データをソート
                sorted_data = dict(sorted(
                    sector_data.items(),
                    key=lambda item: item[1] if item[1] is not None else -float('inf'),
                    reverse=True
                ))
                
                future = loop.run_in_executor(
                    executor,
                    self.sector_generator.generate_sector_performance_chart,
                    sorted_data,
                    filename
                )
                
                return await future
    
    async def generate_multiple_chart_types_async(
        self, 
        chart_data: Dict[str, Any], 
        chart_types: List[str] = None
    ) -> Dict[str, Any]:
        """複数種類のチャートを並行生成"""
        
        if chart_types is None:
            chart_types = ["intraday", "longterm", "sector"]
        
        start_time = time.time()
        self.logger.info(f"Starting async generation for chart types: {chart_types}")
        
        tasks = []
        
        # 各チャートタイプのタスクを作成
        for chart_type in chart_types:
            if chart_type == "intraday":
                task = self._generate_all_intraday_charts_async(chart_data)
            elif chart_type == "longterm":
                task = self._generate_all_longterm_charts_async(chart_data)
            elif chart_type == "sector":
                task = self._generate_sector_chart_wrapper_async(chart_data)
            else:
                self.logger.warning(f"Unknown chart type: {chart_type}")
                continue
            
            tasks.append(task)
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        all_charts = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Chart type {chart_types[i]} generation failed: {result}")
                self.error_handler.handle_error(result, {'chart_type': chart_types[i]})
            elif result:
                if isinstance(result, dict):
                    all_charts.update(result)
                else:
                    all_charts[chart_types[i]] = result
        
        execution_time = time.time() - start_time
        self.logger.info(f"Multiple chart types generation completed in {execution_time:.2f} seconds")
        
        return all_charts
    
    async def _generate_all_intraday_charts_async(
        self, 
        chart_data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """すべてのイントラデイチャートを非同期生成"""
        
        tasks = []
        for ticker_name, data_set in chart_data.items():
            if isinstance(data_set, dict) and "intraday" in data_set:
                if data_set["intraday"] is not None:
                    task = self._generate_intraday_chart_async(
                        ticker_name, data_set["intraday"], 'interactive'
                    )
                    tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        charts = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Intraday chart generation failed: {result}")
            elif result:
                charts.append(result)
        
        return {"Intraday": charts}
    
    async def _generate_all_longterm_charts_async(
        self, 
        chart_data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """すべての長期チャートを非同期生成"""
        
        tasks = []
        for ticker_name, data_set in chart_data.items():
            if isinstance(data_set, dict) and "longterm" in data_set:
                if data_set["longterm"] is not None:
                    task = self._generate_longterm_chart_async(
                        ticker_name, data_set["longterm"], 'interactive'
                    )
                    tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        charts = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Long-term chart generation failed: {result}")
            elif result:
                charts.append(result)
        
        return {"Long-Term": charts}
    
    async def _generate_sector_chart_wrapper_async(
        self, 
        chart_data: Dict[str, Any]
    ) -> Optional[str]:
        """セクターチャート生成のラッパー"""
        
        # セクターデータを抽出
        sector_data = chart_data.get("sector_performance", {})
        
        if not sector_data:
            self.logger.warning("No sector data found for chart generation")
            return None
        
        return await self.generate_sector_chart_async(sector_data)
    
    async def generate_chart_batch_async(
        self, 
        chart_requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """チャート生成リクエストをバッチ処理"""
        
        start_time = time.time()
        self.logger.info(f"Starting batch chart generation for {len(chart_requests)} requests")
        
        # リクエストごとにタスクを作成
        tasks = []
        for request in chart_requests:
            chart_type = request.get('type', 'candlestick')
            
            if chart_type == 'candlestick':
                task = self._generate_candlestick_chart_from_request_async(request)
            elif chart_type == 'sector':
                task = self._generate_sector_chart_from_request_async(request)
            else:
                self.logger.warning(f"Unknown chart type in request: {chart_type}")
                continue
            
            tasks.append(task)
        
        # タスクを並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を整理
        batch_results = []
        for i, result in enumerate(results):
            request_id = chart_requests[i].get('id', f'request_{i}')
            if isinstance(result, Exception):
                self.logger.error(f"Batch chart generation failed for {request_id}: {result}")
                self.error_handler.handle_error(result, {'request_id': request_id})
                batch_results.append((request_id, None))
            else:
                batch_results.append((request_id, result))
        
        execution_time = time.time() - start_time
        self.logger.info(f"Batch chart generation completed in {execution_time:.2f} seconds")
        
        return batch_results
    
    async def _generate_candlestick_chart_from_request_async(
        self, 
        request: Dict[str, Any]
    ) -> Optional[str]:
        """リクエストからキャンドルスティックチャートを生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = request.get('data')
                title = request.get('title', 'Chart')
                filename = request.get('filename', 'chart.html')
                chart_type = request.get('chart_type', 'interactive')
                
                if chart_type == 'interactive':
                    future = loop.run_in_executor(
                        executor,
                        self.candlestick_generator.generate_interactive_chart,
                        data,
                        title,
                        filename
                    )
                else:
                    future = loop.run_in_executor(
                        executor,
                        self.candlestick_generator.generate_static_chart,
                        data,
                        title,
                        filename
                    )
                
                return await future
    
    async def _generate_sector_chart_from_request_async(
        self, 
        request: Dict[str, Any]
    ) -> Optional[str]:
        """リクエストからセクターチャートを生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = request.get('data')
                filename = request.get('filename', 'sector_chart.html')
                
                future = loop.run_in_executor(
                    executor,
                    self.sector_generator.generate_sector_performance_chart,
                    data,
                    filename
                )
                
                return await future
    
    async def generate_with_timeout(
        self, 
        coro, 
        timeout: float = 60.0
    ) -> Any:
        """タイムアウト付きでチャート生成を実行"""
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Chart generation timed out after {timeout} seconds")
            raise ChartGenerationError(f"Chart generation timed out after {timeout} seconds")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクスを取得"""
        
        return {
            "max_workers": self.max_workers,
            "max_concurrent_charts": self.max_concurrent_charts,
            "charts_dir": str(self.charts_dir),
            "error_summary": self.error_handler.get_error_summary(),
            "candlestick_errors": self.candlestick_generator.get_error_summary(),
            "sector_errors": self.sector_generator.get_error_summary()
        }
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        
        # エラー履歴をクリア
        self.error_handler.clear_history()
        
        # 各ジェネレーターのエラー履歴をクリア
        if hasattr(self.candlestick_generator, 'clear_error_history'):
            self.candlestick_generator.clear_error_history()
        if hasattr(self.sector_generator, 'clear_error_history'):
            self.sector_generator.clear_error_history()
        
        self.logger.info("AsyncChartGenerator cleanup completed")


# 使用例関数
async def generate_all_charts_async(chart_data: Dict[str, Any], charts_dir: str = "charts"):
    """すべてのチャートを非同期で生成する使用例"""
    
    async_generator = AsyncChartGenerator(charts_dir)
    
    try:
        # すべてのチャートを生成
        charts = await async_generator.generate_all_charts(chart_data)
        
        # セクターチャートも生成
        if "sector_performance" in chart_data:
            sector_chart = await async_generator.generate_sector_chart_async(
                chart_data["sector_performance"]
            )
            if sector_chart:
                charts["sector"] = sector_chart
        
        return charts
        
    except Exception as e:
        logging.error(f"Error in generate_all_charts_async: {e}")
        raise
    
    finally:
        async_generator.cleanup()


async def generate_specific_chart_types(
    chart_data: Dict[str, Any], 
    chart_types: List[str] = None,
    charts_dir: str = "charts"
):
    """特定のチャートタイプを非同期で生成"""
    
    async_generator = AsyncChartGenerator(charts_dir)
    
    try:
        charts = await async_generator.generate_multiple_chart_types_async(
            chart_data, chart_types
        )
        return charts
        
    except Exception as e:
        logging.error(f"Error in generate_specific_chart_types: {e}")
        raise
    
    finally:
        async_generator.cleanup()


# 同期インターフェース（後方互換性のため）
def generate_charts_sync(chart_data: Dict[str, Any], charts_dir: str = "charts") -> Dict[str, Any]:
    """同期インターフェースでチャートを生成"""
    
    async def _generate():
        async_generator = AsyncChartGenerator(charts_dir)
        try:
            return await async_generator.generate_all_charts(chart_data)
        finally:
            async_generator.cleanup()
    
    return asyncio.run(_generate())


def generate_sector_chart_sync(
    sector_data: Dict[str, Any], 
    filename: str = "sector_performance_chart.html",
    charts_dir: str = "charts"
) -> Optional[str]:
    """同期インターフェースでセクターチャートを生成"""
    
    async def _generate():
        async_generator = AsyncChartGenerator(charts_dir)
        try:
            return await async_generator.generate_sector_chart_async(sector_data, filename)
        finally:
            async_generator.cleanup()
    
    return asyncio.run(_generate())