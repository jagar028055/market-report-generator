"""
非同期レポート生成クラス
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from pathlib import Path

from .async_data_fetcher import AsyncDataFetcher
from .async_chart_generator import AsyncChartGenerator
from ..config import get_system_config
from ..utils.exceptions import ReportGenerationError
from ..utils.error_handler import ErrorHandler


class AsyncReportGenerator:
    """非同期レポート生成クラス"""
    
    def __init__(self, 
                 charts_dir: str = "charts",
                 reports_dir: str = "reports", 
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.charts_dir = Path(charts_dir)
        self.reports_dir = Path(reports_dir)
        self.system_config = get_system_config()
        self.error_handler = ErrorHandler(self.logger)
        
        # 非同期コンポーネントを初期化
        self.data_fetcher = AsyncDataFetcher(self.logger)
        self.chart_generator = AsyncChartGenerator(str(self.charts_dir), self.logger)
        
        # 並行処理設定
        self.max_workers = self.system_config.MAX_WORKERS
        self.max_concurrent_operations = self.system_config.MAX_CONCURRENT_REQUESTS
        
        # セマフォを作成
        self.semaphore = asyncio.Semaphore(self.max_concurrent_operations)
        
        # 出力ディレクトリを作成
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized AsyncReportGenerator with {self.max_workers} workers")
    
    async def generate_complete_report(self, **kwargs) -> Dict[str, Any]:
        """完全なレポートを非同期で生成"""
        
        start_time = time.time()
        self.logger.info("Starting complete report generation")
        
        try:
            # フェーズ1: データ取得とチャート生成を並行実行
            data_task = self.data_fetcher.fetch_all_data(**kwargs)
            
            # データ取得の完了を待つ
            market_data = await data_task
            
            # フェーズ2: チャートデータ取得と基本チャート生成を並行実行
            chart_data_task = self.data_fetcher.fetch_chart_data_async()
            basic_charts_task = self.chart_generator.generate_all_charts(market_data)
            
            # セクターチャートが必要な場合
            sector_chart_task = None
            if "sector_performance" in market_data:
                sector_chart_task = self.chart_generator.generate_sector_chart_async(
                    market_data["sector_performance"]
                )
            
            # 並行実行
            tasks = [chart_data_task, basic_charts_task]
            if sector_chart_task:
                tasks.append(sector_chart_task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を整理
            chart_data = results[0] if not isinstance(results[0], Exception) else {}
            basic_charts = results[1] if not isinstance(results[1], Exception) else {}
            sector_chart = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
            
            # フェーズ3: チャートデータからの追加チャート生成
            additional_charts = {}
            if chart_data:
                additional_charts = await self.chart_generator.generate_all_charts(chart_data)
            
            # フェーズ4: レポート構築
            report = await self._build_report_async(
                market_data, 
                basic_charts, 
                additional_charts,
                sector_chart
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"Complete report generation completed in {execution_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Complete report generation failed: {e}")
            self.error_handler.handle_error(e, {'operation': 'complete_report_generation'})
            raise ReportGenerationError(f"Failed to generate complete report: {e}")
    
    async def generate_data_only_report(self, **kwargs) -> Dict[str, Any]:
        """データのみのレポートを非同期で生成"""
        
        start_time = time.time()
        self.logger.info("Starting data-only report generation")
        
        try:
            # データ取得のみ実行
            market_data = await self.data_fetcher.fetch_all_data(**kwargs)
            
            # データサマリーを作成
            report = await self._build_data_summary_async(market_data)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Data-only report generation completed in {execution_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Data-only report generation failed: {e}")
            self.error_handler.handle_error(e, {'operation': 'data_only_report_generation'})
            raise ReportGenerationError(f"Failed to generate data-only report: {e}")
    
    async def generate_charts_only_report(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """チャートのみのレポートを非同期で生成"""
        
        start_time = time.time()
        self.logger.info("Starting charts-only report generation")
        
        try:
            # チャート生成タスクを並行実行
            tasks = [
                self.chart_generator.generate_all_charts(chart_data),
                self.chart_generator.generate_sector_chart_async(
                    chart_data.get("sector_performance", {})
                )
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を整理
            charts = results[0] if not isinstance(results[0], Exception) else {}
            sector_chart = results[1] if not isinstance(results[1], Exception) else None
            
            # チャートサマリーを作成
            report = await self._build_charts_summary_async(charts, sector_chart)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Charts-only report generation completed in {execution_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Charts-only report generation failed: {e}")
            self.error_handler.handle_error(e, {'operation': 'charts_only_report_generation'})
            raise ReportGenerationError(f"Failed to generate charts-only report: {e}")
    
    async def generate_custom_report(self, 
                                   components: List[str], 
                                   **kwargs) -> Dict[str, Any]:
        """カスタムレポートを非同期で生成"""
        
        start_time = time.time()
        self.logger.info(f"Starting custom report generation with components: {components}")
        
        try:
            # 必要なコンポーネントに基づいてタスクを作成
            tasks = []
            
            if "data" in components:
                tasks.append(self._generate_data_component_async(**kwargs))
            
            if "charts" in components:
                if "data" in components:
                    # データが必要な場合は、データ取得後にチャート生成
                    data_task = self._generate_data_component_async(**kwargs)
                    data_result = await data_task
                    chart_task = self._generate_charts_component_async(data_result)
                    tasks.append(chart_task)
                else:
                    # データが不要な場合は、提供されたデータでチャート生成
                    chart_data = kwargs.get('chart_data', {})
                    tasks.append(self._generate_charts_component_async(chart_data))
            
            if "analytics" in components:
                tasks.append(self._generate_analytics_component_async(**kwargs))
            
            # タスクを並行実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を統合
            report = await self._build_custom_report_async(components, results)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Custom report generation completed in {execution_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Custom report generation failed: {e}")
            self.error_handler.handle_error(e, {'operation': 'custom_report_generation'})
            raise ReportGenerationError(f"Failed to generate custom report: {e}")
    
    async def _generate_data_component_async(self, **kwargs) -> Dict[str, Any]:
        """データコンポーネントを非同期で生成"""
        
        async with self.semaphore:
            return await self.data_fetcher.fetch_all_data(**kwargs)
    
    async def _generate_charts_component_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """チャートコンポーネントを非同期で生成"""
        
        async with self.semaphore:
            # 基本チャート生成
            basic_charts = await self.chart_generator.generate_all_charts(data)
            
            # セクターチャート生成
            sector_chart = None
            if "sector_performance" in data:
                sector_chart = await self.chart_generator.generate_sector_chart_async(
                    data["sector_performance"]
                )
            
            return {
                "basic_charts": basic_charts,
                "sector_chart": sector_chart
            }
    
    async def _generate_analytics_component_async(self, **kwargs) -> Dict[str, Any]:
        """分析コンポーネントを非同期で生成"""
        
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # 分析処理をスレッドプールで実行
                future = loop.run_in_executor(
                    executor,
                    self._perform_analytics,
                    kwargs
                )
                
                return await future
    
    def _perform_analytics(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """分析処理を実行"""
        
        # 基本的な分析処理
        analytics = {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "basic",
            "performance_metrics": self.get_performance_metrics()
        }
        
        return analytics
    
    async def _build_report_async(self, 
                                market_data: Dict[str, Any],
                                basic_charts: Dict[str, Any],
                                additional_charts: Dict[str, Any],
                                sector_chart: Optional[str]) -> Dict[str, Any]:
        """レポートを構築"""
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._build_report_sync,
                market_data,
                basic_charts,
                additional_charts,
                sector_chart
            )
            
            return await future
    
    def _build_report_sync(self,
                          market_data: Dict[str, Any],
                          basic_charts: Dict[str, Any],
                          additional_charts: Dict[str, Any],
                          sector_chart: Optional[str]) -> Dict[str, Any]:
        """同期的にレポートを構築"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data": market_data,
            "charts": {
                "basic": basic_charts,
                "additional": additional_charts,
                "sector": sector_chart
            },
            "summary": {
                "data_sources": len(market_data),
                "chart_count": len(basic_charts.get("Intraday", [])) + len(basic_charts.get("Long-Term", [])),
                "has_sector_chart": sector_chart is not None
            },
            "metadata": {
                "generator": "AsyncReportGenerator",
                "version": "1.0"
            }
        }
        
        return report
    
    async def _build_data_summary_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """データサマリーを構築"""
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._build_data_summary_sync,
                market_data
            )
            
            return await future
    
    def _build_data_summary_sync(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """同期的にデータサマリーを構築"""
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "data": market_data,
            "summary": {
                "data_sources": len(market_data),
                "market_data_available": "market_data" in market_data,
                "news_articles": len(market_data.get("news_articles", [])),
                "economic_indicators": len(market_data.get("economic_indicators", {}).get("today_scheduled", []))
            },
            "metadata": {
                "generator": "AsyncReportGenerator",
                "type": "data_only",
                "version": "1.0"
            }
        }
        
        return summary
    
    async def _build_charts_summary_async(self, 
                                        charts: Dict[str, Any],
                                        sector_chart: Optional[str]) -> Dict[str, Any]:
        """チャートサマリーを構築"""
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._build_charts_summary_sync,
                charts,
                sector_chart
            )
            
            return await future
    
    def _build_charts_summary_sync(self, 
                                  charts: Dict[str, Any],
                                  sector_chart: Optional[str]) -> Dict[str, Any]:
        """同期的にチャートサマリーを構築"""
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "charts": {
                "basic": charts,
                "sector": sector_chart
            },
            "summary": {
                "intraday_charts": len(charts.get("Intraday", [])),
                "longterm_charts": len(charts.get("Long-Term", [])),
                "has_sector_chart": sector_chart is not None
            },
            "metadata": {
                "generator": "AsyncReportGenerator",
                "type": "charts_only",
                "version": "1.0"
            }
        }
        
        return summary
    
    async def _build_custom_report_async(self, 
                                       components: List[str],
                                       results: List[Any]) -> Dict[str, Any]:
        """カスタムレポートを構築"""
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._build_custom_report_sync,
                components,
                results
            )
            
            return await future
    
    def _build_custom_report_sync(self, 
                                 components: List[str],
                                 results: List[Any]) -> Dict[str, Any]:
        """同期的にカスタムレポートを構築"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "components": components,
            "metadata": {
                "generator": "AsyncReportGenerator",
                "type": "custom",
                "version": "1.0"
            }
        }
        
        # 結果を各コンポーネントに割り当て
        for i, component in enumerate(components):
            if i < len(results) and not isinstance(results[i], Exception):
                report[component] = results[i]
            else:
                report[component] = None
        
        return report
    
    async def generate_with_timeout(self, 
                                  coro, 
                                  timeout: float = 300.0) -> Any:
        """タイムアウト付きでレポート生成を実行"""
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Report generation timed out after {timeout} seconds")
            raise ReportGenerationError(f"Report generation timed out after {timeout} seconds")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクスを取得"""
        
        return {
            "max_workers": self.max_workers,
            "max_concurrent_operations": self.max_concurrent_operations,
            "charts_dir": str(self.charts_dir),
            "reports_dir": str(self.reports_dir),
            "error_summary": self.error_handler.get_error_summary(),
            "data_fetcher_metrics": self.data_fetcher.get_performance_metrics(),
            "chart_generator_metrics": self.chart_generator.get_performance_metrics()
        }
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        
        # エラー履歴をクリア
        self.error_handler.clear_history()
        
        # 各コンポーネントのクリーンアップ
        self.data_fetcher.cleanup()
        self.chart_generator.cleanup()
        
        self.logger.info("AsyncReportGenerator cleanup completed")


# 使用例関数
async def generate_complete_market_report(**kwargs):
    """完全な市場レポートを非同期で生成する使用例"""
    
    report_generator = AsyncReportGenerator()
    
    try:
        report = await report_generator.generate_complete_report(**kwargs)
        return report
        
    except Exception as e:
        logging.error(f"Error in generate_complete_market_report: {e}")
        raise
    
    finally:
        report_generator.cleanup()


async def generate_quick_data_report(**kwargs):
    """クイックデータレポートを非同期で生成"""
    
    report_generator = AsyncReportGenerator()
    
    try:
        report = await report_generator.generate_data_only_report(**kwargs)
        return report
        
    except Exception as e:
        logging.error(f"Error in generate_quick_data_report: {e}")
        raise
    
    finally:
        report_generator.cleanup()


# 同期インターフェース（後方互換性のため）
def generate_report_sync(**kwargs) -> Dict[str, Any]:
    """同期インターフェースでレポートを生成"""
    
    async def _generate():
        report_generator = AsyncReportGenerator()
        try:
            return await report_generator.generate_complete_report(**kwargs)
        finally:
            report_generator.cleanup()
    
    return asyncio.run(_generate())


def generate_data_report_sync(**kwargs) -> Dict[str, Any]:
    """同期インターフェースでデータレポートを生成"""
    
    async def _generate():
        report_generator = AsyncReportGenerator()
        try:
            return await report_generator.generate_data_only_report(**kwargs)
        finally:
            report_generator.cleanup()
    
    return asyncio.run(_generate())