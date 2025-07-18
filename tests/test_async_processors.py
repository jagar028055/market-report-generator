"""
非同期処理のテスト
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import logging
from datetime import datetime
import tempfile
import pytest

from src.async_processors.async_data_fetcher import AsyncDataFetcher
from src.async_processors.async_chart_generator import AsyncChartGenerator
from src.async_processors.async_report_generator import AsyncReportGenerator
from src.async_processors.task_manager import TaskManager, TaskStatus, TaskPriority, TaskResult
from src.utils.exceptions import DataFetchError, ChartGenerationError, ReportGenerationError


class AsyncTestCase(unittest.TestCase):
    """非同期テスト用の基底クラス"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger = Mock(spec=logging.Logger)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.loop.close()
    
    def run_async(self, coro):
        """非同期関数を実行"""
        return self.loop.run_until_complete(coro)


class TestAsyncDataFetcher(AsyncTestCase):
    """AsyncDataFetcherのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        super().setUp()
        self.fetcher = AsyncDataFetcher(self.logger)
        
        # モックデータ
        self.mock_market_data = {
            'S&P500': {'price': 4500, 'change': 1.5},
            'NASDAQ': {'price': 15000, 'change': 2.0}
        }
        
        self.mock_news_data = [
            {'title': 'Test News 1', 'summary': 'Test summary 1'},
            {'title': 'Test News 2', 'summary': 'Test summary 2'}
        ]
        
        self.mock_economic_data = {
            'yesterday': [],
            'today_scheduled': []
        }
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.fetcher.cleanup()
        super().tearDown()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.fetcher.logger)
        self.assertIsNotNone(self.fetcher.system_config)
        self.assertIsNotNone(self.fetcher.error_handler)
        self.assertIsNotNone(self.fetcher.market_fetcher)
        self.assertIsNotNone(self.fetcher.news_fetcher)
        self.assertIsNotNone(self.fetcher.economic_fetcher)
    
    @patch('src.async_processors.async_data_fetcher.ThreadPoolExecutor')
    def test_fetch_market_data_async(self, mock_executor):
        """市場データの非同期取得テスト"""
        mock_future = Mock()
        mock_future.result.return_value = self.mock_market_data
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        with patch.object(self.fetcher.market_fetcher, 'get_market_data') as mock_get_data:
            mock_get_data.return_value = self.mock_market_data
            
            result = self.run_async(self.fetcher._fetch_market_data_async())
            
            self.assertEqual(result, self.mock_market_data)
    
    @patch('src.async_processors.async_data_fetcher.ThreadPoolExecutor')
    def test_fetch_news_data_async(self, mock_executor):
        """ニュースデータの非同期取得テスト"""
        mock_future = Mock()
        mock_future.result.return_value = self.mock_news_data
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        with patch.object(self.fetcher.news_fetcher, 'fetch_data') as mock_fetch_data:
            mock_fetch_data.return_value = self.mock_news_data
            
            result = self.run_async(self.fetcher._fetch_news_data_async())
            
            self.assertEqual(result, self.mock_news_data)
    
    @patch('src.async_processors.async_data_fetcher.ThreadPoolExecutor')
    def test_fetch_economic_data_async(self, mock_executor):
        """経済データの非同期取得テスト"""
        mock_future = Mock()
        mock_future.result.return_value = self.mock_economic_data
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        with patch.object(self.fetcher.economic_fetcher, 'get_economic_indicators') as mock_get_indicators:
            mock_get_indicators.return_value = self.mock_economic_data
            
            result = self.run_async(self.fetcher._fetch_economic_data_async())
            
            self.assertEqual(result, self.mock_economic_data)
    
    @patch('src.async_processors.async_data_fetcher.ThreadPoolExecutor')
    def test_fetch_all_data(self, mock_executor):
        """すべてのデータの非同期取得テスト"""
        # 各フェッチャーをモック
        with patch.object(self.fetcher, '_fetch_market_data_async') as mock_market, \
             patch.object(self.fetcher, '_fetch_economic_data_async') as mock_economic, \
             patch.object(self.fetcher, '_fetch_sector_data_async') as mock_sector, \
             patch.object(self.fetcher, '_fetch_news_data_async') as mock_news:
            
            mock_market.return_value = self.mock_market_data
            mock_economic.return_value = self.mock_economic_data
            mock_sector.return_value = {}
            mock_news.return_value = self.mock_news_data
            
            result = self.run_async(self.fetcher.fetch_all_data())
            
            self.assertIsInstance(result, dict)
            self.assertIn('market_data', result)
            self.assertIn('economic_indicators', result)
            self.assertIn('sector_performance', result)
            self.assertIn('news_articles', result)
    
    def test_fetch_with_timeout(self):
        """タイムアウト付き取得のテスト"""
        async def slow_operation():
            await asyncio.sleep(2)
            return "result"
        
        # タイムアウトエラーが発生することを確認
        with self.assertRaises(Exception):
            self.run_async(self.fetcher.fetch_with_timeout(slow_operation(), timeout=0.1))
    
    def test_get_performance_metrics(self):
        """パフォーマンスメトリクス取得のテスト"""
        metrics = self.fetcher.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('max_workers', metrics)
        self.assertIn('max_concurrent_requests', metrics)
        self.assertIn('error_summary', metrics)
    
    def test_cleanup(self):
        """クリーンアップのテスト"""
        # クリーンアップが例外を発生させないことを確認
        try:
            self.fetcher.cleanup()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")


class TestAsyncChartGenerator(AsyncTestCase):
    """AsyncChartGeneratorのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.generator = AsyncChartGenerator(self.temp_dir, self.logger)
        
        # モックチャートデータ
        self.mock_chart_data = {
            'S&P500': {
                'intraday': pd.DataFrame({
                    'Open': [100, 101], 'High': [105, 106], 
                    'Low': [95, 96], 'Close': [102, 103]
                }),
                'longterm': pd.DataFrame({
                    'Open': [100, 101], 'High': [105, 106], 
                    'Low': [95, 96], 'Close': [102, 103]
                })
            }
        }
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.generator.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.logger)
        self.assertIsNotNone(self.generator.system_config)
        self.assertIsNotNone(self.generator.error_handler)
        self.assertIsNotNone(self.generator.candlestick_generator)
        self.assertIsNotNone(self.generator.sector_generator)
    
    @patch('src.async_processors.async_chart_generator.ThreadPoolExecutor')
    def test_generate_intraday_chart_async(self, mock_executor):
        """イントラデイチャートの非同期生成テスト"""
        mock_future = Mock()
        mock_future.result.return_value = f"{self.temp_dir}/test_intraday.html"
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        with patch.object(self.generator.candlestick_generator, 'generate_intraday_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/test_intraday.html"
            
            result = self.run_async(self.generator._generate_intraday_chart_async(
                "S&P500", self.mock_chart_data['S&P500']['intraday']
            ))
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['type'], 'intraday')
            self.assertEqual(result['name'], 'S&P500')
    
    @patch('src.async_processors.async_chart_generator.ThreadPoolExecutor')
    def test_generate_longterm_chart_async(self, mock_executor):
        """長期チャートの非同期生成テスト"""
        mock_future = Mock()
        mock_future.result.return_value = f"{self.temp_dir}/test_longterm.html"
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        with patch.object(self.generator.candlestick_generator, 'generate_longterm_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/test_longterm.html"
            
            result = self.run_async(self.generator._generate_longterm_chart_async(
                "S&P500", self.mock_chart_data['S&P500']['longterm']
            ))
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['type'], 'longterm')
            self.assertEqual(result['name'], 'S&P500')
    
    @patch('src.async_processors.async_chart_generator.ThreadPoolExecutor')
    def test_generate_sector_chart_async(self, mock_executor):
        """セクターチャートの非同期生成テスト"""
        mock_future = Mock()
        mock_future.result.return_value = f"{self.temp_dir}/sector_chart.html"
        
        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
        
        sector_data = {'Technology': 2.5, 'Healthcare': 1.8}
        
        with patch.object(self.generator.sector_generator, 'generate_sector_performance_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/sector_chart.html"
            
            result = self.run_async(self.generator.generate_sector_chart_async(sector_data))
            
            self.assertIsNotNone(result)
    
    def test_generate_with_timeout(self):
        """タイムアウト付きチャート生成のテスト"""
        async def slow_chart_generation():
            await asyncio.sleep(2)
            return "chart_path"
        
        # タイムアウトエラーが発生することを確認
        with self.assertRaises(Exception):
            self.run_async(self.generator.generate_with_timeout(slow_chart_generation(), timeout=0.1))
    
    def test_get_performance_metrics(self):
        """パフォーマンスメトリクス取得のテスト"""
        metrics = self.generator.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('max_workers', metrics)
        self.assertIn('max_concurrent_charts', metrics)
        self.assertIn('charts_dir', metrics)


class TestAsyncReportGenerator(AsyncTestCase):
    """AsyncReportGeneratorのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.generator = AsyncReportGenerator(logger=self.logger)
        
        # モックデータ
        self.mock_data = {
            'market_data': {'S&P500': {'price': 4500}},
            'news_articles': [{'title': 'Test News'}],
            'economic_indicators': {'yesterday': [], 'today_scheduled': []}
        }
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.generator.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.logger)
        self.assertIsNotNone(self.generator.system_config)
        self.assertIsNotNone(self.generator.error_handler)
        self.assertIsNotNone(self.generator.data_fetcher)
        self.assertIsNotNone(self.generator.chart_generator)
    
    def test_generate_data_only_report(self):
        """データのみレポート生成のテスト"""
        with patch.object(self.generator.data_fetcher, 'fetch_all_data') as mock_fetch:
            mock_fetch.return_value = self.mock_data
            
            result = self.run_async(self.generator.generate_data_only_report())
            
            self.assertIsInstance(result, dict)
            self.assertIn('generated_at', result)
            self.assertIn('data', result)
            self.assertIn('summary', result)
            self.assertIn('metadata', result)
    
    def test_generate_with_timeout(self):
        """タイムアウト付きレポート生成のテスト"""
        async def slow_report_generation():
            await asyncio.sleep(2)
            return {"report": "data"}
        
        # タイムアウトエラーが発生することを確認
        with self.assertRaises(Exception):
            self.run_async(self.generator.generate_with_timeout(slow_report_generation(), timeout=0.1))
    
    def test_get_performance_metrics(self):
        """パフォーマンスメトリクス取得のテスト"""
        metrics = self.generator.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('max_workers', metrics)
        self.assertIn('max_concurrent_operations', metrics)
        self.assertIn('data_fetcher_metrics', metrics)
        self.assertIn('chart_generator_metrics', metrics)


class TestTaskManager(AsyncTestCase):
    """TaskManagerのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        super().setUp()
        self.manager = TaskManager(self.logger)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.manager.cleanup()
        super().tearDown()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.manager.logger)
        self.assertIsNotNone(self.manager.system_config)
        self.assertIsNotNone(self.manager.error_handler)
        self.assertIsNotNone(self.manager.data_fetcher)
        self.assertIsNotNone(self.manager.chart_generator)
        self.assertIsNotNone(self.manager.report_generator)
    
    def test_add_task(self):
        """タスク追加のテスト"""
        async def test_task():
            return "result"
        
        task_id = self.run_async(self.manager.add_task(
            "test_task", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.manager.tasks)
        self.assertIn(task_id, self.manager.task_results)
    
    def test_add_data_fetch_task(self):
        """データ取得タスク追加のテスト"""
        task_id = self.run_async(self.manager.add_data_fetch_task(
            "fetch_data", 
            TaskPriority.HIGH
        ))
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.manager.tasks)
        
        # タスクのメタデータを確認
        task = self.manager.tasks[task_id]
        self.assertEqual(task.metadata['type'], 'data_fetch')
    
    def test_add_chart_generation_task(self):
        """チャート生成タスク追加のテスト"""
        chart_data = {'S&P500': {'intraday': pd.DataFrame()}}
        
        task_id = self.run_async(self.manager.add_chart_generation_task(
            chart_data, 
            "generate_charts", 
            TaskPriority.MEDIUM
        ))
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.manager.tasks)
        
        # タスクのメタデータを確認
        task = self.manager.tasks[task_id]
        self.assertEqual(task.metadata['type'], 'chart_generation')
    
    def test_add_report_generation_task(self):
        """レポート生成タスク追加のテスト"""
        task_id = self.run_async(self.manager.add_report_generation_task(
            "generate_report", 
            TaskPriority.LOW
        ))
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.manager.tasks)
        
        # タスクのメタデータを確認
        task = self.manager.tasks[task_id]
        self.assertEqual(task.metadata['type'], 'report_generation')
    
    def test_execute_task(self):
        """タスク実行のテスト"""
        async def test_task():
            return "test_result"
        
        task_id = self.run_async(self.manager.add_task(
            "test_task", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        result = self.run_async(self.manager.execute_task(task_id))
        
        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "test_result")
    
    def test_execute_task_with_timeout(self):
        """タイムアウト付きタスク実行のテスト"""
        async def slow_task():
            await asyncio.sleep(2)
            return "result"
        
        task_id = self.run_async(self.manager.add_task(
            "slow_task", 
            slow_task, 
            TaskPriority.HIGH,
            timeout=0.1
        ))
        
        result = self.run_async(self.manager.execute_task(task_id))
        
        self.assertEqual(result.status, TaskStatus.TIMEOUT)
    
    def test_execute_task_with_error(self):
        """エラーが発生するタスク実行のテスト"""
        async def error_task():
            raise ValueError("Test error")
        
        task_id = self.run_async(self.manager.add_task(
            "error_task", 
            error_task, 
            TaskPriority.HIGH
        ))
        
        result = self.run_async(self.manager.execute_task(task_id))
        
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIsInstance(result.error, ValueError)
    
    def test_execute_task_with_dependencies(self):
        """依存関係を持つタスク実行のテスト"""
        async def task1():
            return "result1"
        
        async def task2():
            return "result2"
        
        # 依存関係のないタスクを追加
        task1_id = self.run_async(self.manager.add_task(
            "task1", 
            task1, 
            TaskPriority.HIGH
        ))
        
        # 依存関係を持つタスクを追加
        task2_id = self.run_async(self.manager.add_task(
            "task2", 
            task2, 
            TaskPriority.HIGH,
            dependencies=[task1_id]
        ))
        
        # 依存関係を満たすために最初のタスクを実行
        result1 = self.run_async(self.manager.execute_task(task1_id))
        self.assertEqual(result1.status, TaskStatus.COMPLETED)
        
        # 依存関係を持つタスクを実行
        result2 = self.run_async(self.manager.execute_task(task2_id))
        self.assertEqual(result2.status, TaskStatus.COMPLETED)
    
    def test_get_task_status(self):
        """タスクステータス取得のテスト"""
        async def test_task():
            return "result"
        
        task_id = self.run_async(self.manager.add_task(
            "test_task", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        status = self.manager.get_task_status(task_id)
        self.assertEqual(status, TaskStatus.PENDING)
    
    def test_get_statistics(self):
        """統計情報取得のテスト"""
        async def test_task():
            return "result"
        
        # いくつかのタスクを追加
        task1_id = self.run_async(self.manager.add_task(
            "task1", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        task2_id = self.run_async(self.manager.add_task(
            "task2", 
            test_task, 
            TaskPriority.MEDIUM
        ))
        
        stats = self.manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tasks', stats)
        self.assertIn('completed_tasks', stats)
        self.assertIn('failed_tasks', stats)
        self.assertIn('pending_tasks', stats)
        self.assertIn('success_rate', stats)
        
        self.assertEqual(stats['total_tasks'], 2)
        self.assertEqual(stats['pending_tasks'], 2)
    
    def test_clear_completed_tasks(self):
        """完了タスクのクリアテスト"""
        async def test_task():
            return "result"
        
        task_id = self.run_async(self.manager.add_task(
            "test_task", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        # タスクを実行
        self.run_async(self.manager.execute_task(task_id))
        
        # 完了タスクをクリア
        cleared_count = self.manager.clear_completed_tasks()
        
        self.assertEqual(cleared_count, 1)
        self.assertNotIn(task_id, self.manager.tasks)
    
    def test_cleanup(self):
        """クリーンアップのテスト"""
        async def test_task():
            return "result"
        
        # タスクを追加
        task_id = self.run_async(self.manager.add_task(
            "test_task", 
            test_task, 
            TaskPriority.HIGH
        ))
        
        # クリーンアップを実行
        self.manager.cleanup()
        
        # タスクがクリアされることを確認
        self.assertEqual(len(self.manager.tasks), 0)
        self.assertEqual(len(self.manager.task_results), 0)


class TestAsyncProcessorsIntegration(AsyncTestCase):
    """非同期処理の統合テスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.task_manager = TaskManager(self.logger)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.task_manager.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()
    
    def test_complete_workflow(self):
        """完全なワークフローのテスト"""
        from src.async_processors.task_manager import create_data_only_workflow
        
        # データのみのワークフローを作成
        workflow_tasks = self.run_async(create_data_only_workflow(self.task_manager))
        
        self.assertIsInstance(workflow_tasks, list)
        self.assertGreater(len(workflow_tasks), 0)
        
        # 各タスクがタスクマネージャーに追加されていることを確認
        for task_id in workflow_tasks:
            self.assertIn(task_id, self.task_manager.tasks)
    
    def test_performance_metrics_consistency(self):
        """パフォーマンスメトリクスの一貫性テスト"""
        # 各コンポーネントのメトリクスを取得
        data_fetcher_metrics = self.task_manager.data_fetcher.get_performance_metrics()
        chart_generator_metrics = self.task_manager.chart_generator.get_performance_metrics()
        report_generator_metrics = self.task_manager.report_generator.get_performance_metrics()
        
        # すべてのメトリクスが辞書であることを確認
        self.assertIsInstance(data_fetcher_metrics, dict)
        self.assertIsInstance(chart_generator_metrics, dict)
        self.assertIsInstance(report_generator_metrics, dict)
        
        # 共通のキーが存在することを確認
        common_keys = ['max_workers', 'error_summary']
        for metrics in [data_fetcher_metrics, chart_generator_metrics, report_generator_metrics]:
            for key in common_keys:
                self.assertIn(key, metrics)
    
    def test_error_handling_consistency(self):
        """エラーハンドリングの一貫性テスト"""
        # 各コンポーネントのエラーハンドラーが正しく設定されていることを確認
        components = [
            self.task_manager.data_fetcher,
            self.task_manager.chart_generator,
            self.task_manager.report_generator
        ]
        
        for component in components:
            self.assertIsNotNone(component.error_handler)
            self.assertIsNotNone(component.logger)
    
    def test_cleanup_all_components(self):
        """すべてのコンポーネントのクリーンアップテスト"""
        # すべてのコンポーネントでクリーンアップが例外を発生させないことを確認
        components = [
            self.task_manager.data_fetcher,
            self.task_manager.chart_generator,
            self.task_manager.report_generator
        ]
        
        for component in components:
            try:
                component.cleanup()
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Cleanup failed for {type(component).__name__}: {e}")


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)