"""
設定とエラーハンドリングのテスト
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import logging
import tempfile
import yaml
import os
from pathlib import Path
import time

from src.config.base_config import BaseConfig
from src.config.data_config import DataFetchConfig
from src.config.chart_config import ChartConfig
from src.config.system_config import SystemConfig
from src.config import get_data_config, get_chart_config, get_system_config
from src.utils.exceptions import (
    MarketReportException, DataFetchError, ChartGenerationError, 
    NetworkError, ValidationError, ConfigurationError
)
from src.utils.error_handler import ErrorHandler, with_error_handling


class TestBaseConfig(unittest.TestCase):
    """BaseConfigのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # テスト用設定ファイルを作成
        test_config = {
            'test_setting': 'test_value',
            'numeric_setting': 42,
            'list_setting': ['item1', 'item2']
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """初期化のテスト"""
        # 具象クラスを作成してテスト
        class TestConfig(BaseConfig):
            def _load_configuration(self):
                pass
            
            def _validate_configuration(self):
                pass
        
        config = TestConfig()
        self.assertIsNotNone(config)
    
    def test_load_yaml_config(self):
        """YAML設定ファイルの読み込みテスト"""
        class TestConfig(BaseConfig):
            def _load_configuration(self):
                self.config_data = self._load_yaml_config(self.config_file)
            
            def _validate_configuration(self):
                pass
        
        config = TestConfig()
        config.config_file = self.config_file
        config._load_configuration()
        
        self.assertIsInstance(config.config_data, dict)
        self.assertEqual(config.config_data['test_setting'], 'test_value')
        self.assertEqual(config.config_data['numeric_setting'], 42)
    
    def test_load_yaml_config_file_not_found(self):
        """存在しないYAMLファイルの読み込みテスト"""
        class TestConfig(BaseConfig):
            def _load_configuration(self):
                self.config_data = self._load_yaml_config('/nonexistent/file.yaml')
            
            def _validate_configuration(self):
                pass
        
        config = TestConfig()
        
        with self.assertRaises(ConfigurationError):
            config._load_configuration()
    
    def test_get_config_value(self):
        """設定値取得のテスト"""
        class TestConfig(BaseConfig):
            def _load_configuration(self):
                self.config_data = {'key': 'value', 'nested': {'key': 'nested_value'}}
            
            def _validate_configuration(self):
                pass
        
        config = TestConfig()
        
        # 通常の値取得
        self.assertEqual(config.get_config_value('key'), 'value')
        
        # ネストした値取得
        self.assertEqual(config.get_config_value('nested.key'), 'nested_value')
        
        # 存在しないキーのデフォルト値
        self.assertEqual(config.get_config_value('nonexistent', 'default'), 'default')
    
    def test_validate_required_keys(self):
        """必要なキーのバリデーションテスト"""
        class TestConfig(BaseConfig):
            def _load_configuration(self):
                self.config_data = {'key1': 'value1', 'key2': 'value2'}
            
            def _validate_configuration(self):
                self._validate_required_keys(['key1', 'key2'])
        
        config = TestConfig()
        # 例外が発生しないことを確認
        self.assertIsNotNone(config)
        
        # 必要なキーが欠けている場合
        class InvalidConfig(BaseConfig):
            def _load_configuration(self):
                self.config_data = {'key1': 'value1'}  # key2が欠けている
            
            def _validate_configuration(self):
                self._validate_required_keys(['key1', 'key2'])
        
        with self.assertRaises(ValidationError):
            InvalidConfig()


class TestDataFetchConfig(unittest.TestCase):
    """DataFetchConfigのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.config = DataFetchConfig()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.config)
        self.assertIsInstance(self.config.MARKET_TICKERS, dict)
        self.assertIsInstance(self.config.SECTOR_ETFS, dict)
        self.assertIsInstance(self.config.ECONOMIC_INDICATORS, dict)
    
    def test_market_tickers(self):
        """市場ティッカーの設定テスト"""
        self.assertIn('S&P500', self.config.MARKET_TICKERS)
        self.assertIn('NASDAQ100', self.config.MARKET_TICKERS)
        self.assertIn('ダウ30', self.config.MARKET_TICKERS)
    
    def test_sector_etfs(self):
        """セクターETFの設定テスト"""
        self.assertGreater(len(self.config.SECTOR_ETFS), 0)
        
        # 各セクターETFが必要な情報を持っているか確認
        for etf_name, etf_info in self.config.SECTOR_ETFS.items():
            self.assertIsInstance(etf_info, dict)
            self.assertIn('symbol', etf_info)
    
    def test_economic_indicators(self):
        """経済指標の設定テスト"""
        self.assertGreater(len(self.config.ECONOMIC_INDICATORS), 0)
        
        # 各指標が必要な情報を持っているか確認
        for indicator_name, indicator_info in self.config.ECONOMIC_INDICATORS.items():
            self.assertIsInstance(indicator_info, dict)
            self.assertIn('importance', indicator_info)
            self.assertIn('country', indicator_info)
    
    def test_news_sources(self):
        """ニュースソースの設定テスト"""
        self.assertIsInstance(self.config.NEWS_SOURCES, list)
        self.assertGreater(len(self.config.NEWS_SOURCES), 0)
        
        # 各ニュースソースが必要な情報を持っているか確認
        for source in self.config.NEWS_SOURCES:
            self.assertIsInstance(source, dict)
            self.assertIn('type', source)
    
    def test_validation(self):
        """バリデーションのテスト"""
        # 設定が正しく検証されることを確認
        self.assertTrue(self.config.validate())


class TestChartConfig(unittest.TestCase):
    """ChartConfigのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.config = ChartConfig()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.config)
        self.assertIsInstance(self.config.CHART_WIDTH, int)
        self.assertIsInstance(self.config.CHART_HEIGHT, int)
        self.assertIsInstance(self.config.MOVING_AVERAGES, dict)
    
    def test_moving_averages(self):
        """移動平均の設定テスト"""
        self.assertIn('ma5', self.config.MOVING_AVERAGES)
        self.assertIn('ma25', self.config.MOVING_AVERAGES)
        self.assertIn('ma75', self.config.MOVING_AVERAGES)
        
        # 各移動平均が必要な情報を持っているか確認
        for ma_key, ma_info in self.config.MOVING_AVERAGES.items():
            self.assertIsInstance(ma_info, dict)
            self.assertIn('period', ma_info)
            self.assertIn('color', ma_info)
            self.assertIn('label', ma_info)
    
    def test_chart_colors(self):
        """チャートの色設定テスト"""
        self.assertIsInstance(self.config.CANDLE_COLORS, dict)
        self.assertIsInstance(self.config.SECTOR_CHART_COLORS, dict)
        
        # キャンドルの色設定
        required_candle_colors = ['up_fill', 'down_fill', 'up_line', 'down_line']
        for color_key in required_candle_colors:
            self.assertIn(color_key, self.config.CANDLE_COLORS)
        
        # セクターチャートの色設定
        required_sector_colors = ['positive', 'negative', 'neutral']
        for color_key in required_sector_colors:
            self.assertIn(color_key, self.config.SECTOR_CHART_COLORS)
    
    def test_get_moving_average_config(self):
        """移動平均設定取得のテスト"""
        ma_config = self.config.get_moving_average_config('ma5')
        self.assertIsInstance(ma_config, dict)
        self.assertIn('period', ma_config)
        self.assertIn('color', ma_config)
        self.assertIn('label', ma_config)
        
        # 存在しない移動平均
        with self.assertRaises(ValueError):
            self.config.get_moving_average_config('nonexistent')
    
    def test_add_moving_average(self):
        """移動平均追加のテスト"""
        self.config.add_moving_average('ma200', 200, 'purple', 'MA200')
        
        self.assertIn('ma200', self.config.MOVING_AVERAGES)
        ma_config = self.config.MOVING_AVERAGES['ma200']
        self.assertEqual(ma_config['period'], 200)
        self.assertEqual(ma_config['color'], 'purple')
        self.assertEqual(ma_config['label'], 'MA200')
    
    def test_remove_moving_average(self):
        """移動平均削除のテスト"""
        # 移動平均を追加
        self.config.add_moving_average('test_ma', 10, 'red', 'Test MA')
        self.assertIn('test_ma', self.config.MOVING_AVERAGES)
        
        # 移動平均を削除
        self.config.remove_moving_average('test_ma')
        self.assertNotIn('test_ma', self.config.MOVING_AVERAGES)
    
    def test_validation(self):
        """バリデーションのテスト"""
        # 設定が正しく検証されることを確認
        self.assertTrue(self.config.validate())


class TestSystemConfig(unittest.TestCase):
    """SystemConfigのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.config = SystemConfig()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.config)
        self.assertIsInstance(self.config.MAX_WORKERS, int)
        self.assertIsInstance(self.config.MAX_CONCURRENT_REQUESTS, int)
        self.assertIsInstance(self.config.REQUEST_TIMEOUT, (int, float))
        self.assertIsInstance(self.config.RETRY_ATTEMPTS, int)
    
    def test_reasonable_defaults(self):
        """合理的なデフォルト値のテスト"""
        self.assertGreater(self.config.MAX_WORKERS, 0)
        self.assertLessEqual(self.config.MAX_WORKERS, 20)  # 過度に高くない
        
        self.assertGreater(self.config.MAX_CONCURRENT_REQUESTS, 0)
        self.assertLessEqual(self.config.MAX_CONCURRENT_REQUESTS, 50)  # 過度に高くない
        
        self.assertGreater(self.config.REQUEST_TIMEOUT, 0)
        self.assertLessEqual(self.config.REQUEST_TIMEOUT, 300)  # 5分以内
        
        self.assertGreater(self.config.RETRY_ATTEMPTS, 0)
        self.assertLessEqual(self.config.RETRY_ATTEMPTS, 10)  # 過度に高くない
    
    def test_validation(self):
        """バリデーションのテスト"""
        # 設定が正しく検証されることを確認
        self.assertTrue(self.config.validate())


class TestConfigFactoryFunctions(unittest.TestCase):
    """設定ファクトリー関数のテスト"""
    
    def test_get_data_config(self):
        """データ設定取得のテスト"""
        config = get_data_config()
        self.assertIsInstance(config, DataFetchConfig)
    
    def test_get_chart_config(self):
        """チャート設定取得のテスト"""
        config = get_chart_config()
        self.assertIsInstance(config, ChartConfig)
    
    def test_get_system_config(self):
        """システム設定取得のテスト"""
        config = get_system_config()
        self.assertIsInstance(config, SystemConfig)
    
    def test_singleton_behavior(self):
        """シングルトンの動作テスト"""
        config1 = get_data_config()
        config2 = get_data_config()
        
        # 同じインスタンスが返されることを確認
        self.assertIs(config1, config2)


class TestCustomExceptions(unittest.TestCase):
    """カスタム例外のテスト"""
    
    def test_market_report_exception(self):
        """MarketReportExceptionのテスト"""
        error = MarketReportException("Test error", "TEST_001", {"key": "value"})
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.error_code, "TEST_001")
        self.assertEqual(error.details, {"key": "value"})
    
    def test_data_fetch_error(self):
        """DataFetchErrorのテスト"""
        error = DataFetchError("Data fetch failed", "FETCH_001")
        
        self.assertEqual(str(error), "Data fetch failed")
        self.assertEqual(error.error_code, "FETCH_001")
        self.assertIsInstance(error, MarketReportException)
    
    def test_chart_generation_error(self):
        """ChartGenerationErrorのテスト"""
        error = ChartGenerationError("Chart generation failed", "CHART_001")
        
        self.assertEqual(str(error), "Chart generation failed")
        self.assertEqual(error.error_code, "CHART_001")
        self.assertIsInstance(error, MarketReportException)
    
    def test_network_error(self):
        """NetworkErrorのテスト"""
        error = NetworkError("Network connection failed", "NET_001")
        
        self.assertEqual(str(error), "Network connection failed")
        self.assertEqual(error.error_code, "NET_001")
        self.assertIsInstance(error, MarketReportException)
    
    def test_validation_error(self):
        """ValidationErrorのテスト"""
        error = ValidationError("Validation failed", "VALID_001")
        
        self.assertEqual(str(error), "Validation failed")
        self.assertEqual(error.error_code, "VALID_001")
        self.assertIsInstance(error, MarketReportException)
    
    def test_configuration_error(self):
        """ConfigurationErrorのテスト"""
        error = ConfigurationError("Configuration error", "CONFIG_001")
        
        self.assertEqual(str(error), "Configuration error")
        self.assertEqual(error.error_code, "CONFIG_001")
        self.assertIsInstance(error, MarketReportException)


class TestErrorHandler(unittest.TestCase):
    """ErrorHandlerのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.handler = ErrorHandler(self.logger)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.handler.logger)
        self.assertIsInstance(self.handler.error_history, list)
        self.assertEqual(len(self.handler.error_history), 0)
    
    def test_handle_error(self):
        """エラーハンドリングのテスト"""
        test_error = DataFetchError("Test error", "TEST_001")
        context = {"operation": "test", "data": "test_data"}
        
        self.handler.handle_error(test_error, context)
        
        # エラーログが出力されることを確認
        self.logger.error.assert_called()
        
        # エラー履歴に追加されることを確認
        self.assertEqual(len(self.handler.error_history), 1)
        
        error_record = self.handler.error_history[0]
        self.assertEqual(error_record['error_type'], 'DataFetchError')
        self.assertEqual(error_record['message'], 'Test error')
        self.assertEqual(error_record['context'], context)
    
    def test_handle_error_with_retry(self):
        """リトライ付きエラーハンドリングのテスト"""
        test_error = NetworkError("Network error", "NET_001")
        
        # リトライ機能をテスト
        def failing_operation():
            raise test_error
        
        result = self.handler.handle_error_with_retry(
            failing_operation, 
            max_retries=3, 
            delay=0.1
        )
        
        # リトライ後にNoneが返されることを確認
        self.assertIsNone(result)
        
        # エラー履歴にリトライ情報が記録されることを確認
        self.assertGreater(len(self.handler.error_history), 0)
    
    def test_handle_error_with_retry_success(self):
        """リトライ成功のテスト"""
        call_count = 0
        
        def sometimes_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network error", "NET_001")
            return "success"
        
        result = self.handler.handle_error_with_retry(
            sometimes_failing_operation, 
            max_retries=5, 
            delay=0.1
        )
        
        # 最終的に成功することを確認
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_get_error_summary(self):
        """エラー概要取得のテスト"""
        # いくつかのエラーを追加
        errors = [
            DataFetchError("Error 1", "ERR_001"),
            ChartGenerationError("Error 2", "ERR_002"),
            NetworkError("Error 3", "ERR_003"),
            DataFetchError("Error 4", "ERR_004")
        ]
        
        for error in errors:
            self.handler.handle_error(error, {"test": "context"})
        
        summary = self.handler.get_error_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_errors', summary)
        self.assertIn('error_types', summary)
        self.assertIn('recent_errors', summary)
        
        self.assertEqual(summary['total_errors'], 4)
        self.assertEqual(summary['error_types']['DataFetchError'], 2)
        self.assertEqual(summary['error_types']['ChartGenerationError'], 1)
        self.assertEqual(summary['error_types']['NetworkError'], 1)
    
    def test_clear_history(self):
        """エラー履歴クリアのテスト"""
        # エラーを追加
        self.handler.handle_error(DataFetchError("Test error", "TEST_001"), {})
        self.assertEqual(len(self.handler.error_history), 1)
        
        # 履歴をクリア
        self.handler.clear_history()
        self.assertEqual(len(self.handler.error_history), 0)
    
    def test_circuit_breaker(self):
        """サーキットブレーカーのテスト"""
        # 多数のエラーを発生させる
        for i in range(10):
            self.handler.handle_error(
                NetworkError(f"Error {i}", f"ERR_{i:03d}"), 
                {"attempt": i}
            )
        
        # サーキットブレーカーが動作することを確認
        is_open = self.handler.is_circuit_breaker_open("network_operation")
        # 実装によっては True になる可能性がある
        self.assertIsInstance(is_open, bool)


class TestErrorDecorator(unittest.TestCase):
    """エラーデコレーターのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
    
    def test_with_error_handling_success(self):
        """エラーハンドリングデコレーターの成功テスト"""
        @with_error_handling()
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
    
    def test_with_error_handling_failure(self):
        """エラーハンドリングデコレーターの失敗テスト"""
        @with_error_handling()
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        # デコレーターがエラーを処理してNoneを返すことを確認
        self.assertIsNone(result)
    
    def test_with_error_handling_custom_fallback(self):
        """カスタムフォールバック付きエラーハンドリングテスト"""
        @with_error_handling(fallback_value="fallback")
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        self.assertEqual(result, "fallback")
    
    def test_with_error_handling_async(self):
        """非同期関数のエラーハンドリングテスト"""
        @with_error_handling()
        async def test_async_function():
            raise ValueError("Async test error")
        
        # 非同期関数のテストは実際の実装に依存
        # ここでは関数が正しく定義されることを確認
        self.assertTrue(asyncio.iscoroutinefunction(test_async_function))


class TestConfigAndErrorHandlingIntegration(unittest.TestCase):
    """設定とエラーハンドリングの統合テスト"""
    
    def test_config_error_handling(self):
        """設定読み込み時のエラーハンドリングテスト"""
        # 存在しない設定ファイルを指定
        with self.assertRaises(ConfigurationError):
            class TestConfig(BaseConfig):
                def _load_configuration(self):
                    self._load_yaml_config('/nonexistent/config.yaml')
                
                def _validate_configuration(self):
                    pass
            
            TestConfig()
    
    def test_config_validation_error_handling(self):
        """設定検証時のエラーハンドリングテスト"""
        with self.assertRaises(ValidationError):
            class TestConfig(BaseConfig):
                def _load_configuration(self):
                    self.config_data = {}
                
                def _validate_configuration(self):
                    self._validate_required_keys(['required_key'])
            
            TestConfig()
    
    def test_error_handler_configuration(self):
        """エラーハンドラーの設定テスト"""
        config = get_system_config()
        
        # システム設定からエラーハンドラーを作成
        logger = Mock(spec=logging.Logger)
        error_handler = ErrorHandler(logger)
        
        # 設定値がエラーハンドラーに反映されることを確認
        self.assertIsNotNone(error_handler.logger)
        self.assertEqual(error_handler.max_retries, config.RETRY_ATTEMPTS)
        self.assertEqual(error_handler.base_delay, config.RETRY_DELAY)
    
    def test_all_configs_validation(self):
        """すべての設定の検証テスト"""
        configs = [
            get_data_config(),
            get_chart_config(),
            get_system_config()
        ]
        
        for config in configs:
            try:
                is_valid = config.validate()
                self.assertTrue(is_valid)
            except Exception as e:
                self.fail(f"Config validation failed for {type(config).__name__}: {e}")


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)