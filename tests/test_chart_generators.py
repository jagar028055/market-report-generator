"""
チャートジェネレーターのテスト
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import tempfile
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.chart_generators.base_chart_generator import BaseChartGenerator, ChartGeneratorFactory
from src.chart_generators.candlestick_chart_generator import CandlestickChartGenerator
from src.chart_generators.sector_chart_generator import SectorChartGenerator
from src.utils.exceptions import ChartGenerationError


class TestBaseChartGenerator(unittest.TestCase):
    """BaseChartGeneratorのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.temp_dir = tempfile.mkdtemp()
        
        # 具象クラスを作成してテスト
        class TestChartGenerator(BaseChartGenerator):
            def generate_chart(self, data, title, filename, **kwargs):
                # 簡単なテスト用チャート生成
                return f"{self.charts_dir}/{filename}"
        
        self.generator = TestChartGenerator(self.temp_dir, self.logger)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.logger)
        self.assertIsNotNone(self.generator.error_handler)
        self.assertIsNotNone(self.generator.config)
        self.assertTrue(self.generator.charts_dir.exists())
    
    def test_get_output_path(self):
        """出力パスの取得テスト"""
        filename = "test_chart.html"
        output_path = self.generator.get_output_path(filename)
        
        self.assertIsInstance(output_path, Path)
        self.assertEqual(output_path.name, filename)
        self.assertTrue(str(output_path).startswith(self.temp_dir))
    
    def test_validate_data_empty(self):
        """空データのバリデーションテスト"""
        self.assertFalse(self.generator.validate_data(None, "test"))
        self.assertFalse(self.generator.validate_data(pd.DataFrame(), "test"))
        self.assertFalse(self.generator.validate_data({}, "test"))
    
    def test_validate_data_valid(self):
        """有効データのバリデーションテスト"""
        # 有効なDataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.assertTrue(self.generator.validate_data(df, "test"))
        
        # 有効な辞書
        data_dict = {"key": "value"}
        self.assertTrue(self.generator.validate_data(data_dict, "test"))
    
    def test_set_output_directory(self):
        """出力ディレクトリの設定テスト"""
        new_dir = tempfile.mkdtemp()
        self.generator.set_output_directory(new_dir)
        
        self.assertEqual(str(self.generator.charts_dir), new_dir)
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        test_error = ChartGenerationError("Test error")
        self.generator._handle_generation_error(test_error, "test operation")
        
        self.generator.logger.error.assert_called()


class TestCandlestickChartGenerator(unittest.TestCase):
    """CandlestickChartGeneratorのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.temp_dir = tempfile.mkdtemp()
        self.generator = CandlestickChartGenerator(self.temp_dir, self.logger)
        
        # モックOHLCデータ
        self.mock_ohlc_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.logger)
        self.assertIsNotNone(self.generator.error_handler)
        self.assertIsNotNone(self.generator.config)
    
    def test_validate_candlestick_data_valid(self):
        """有効なキャンドルスティックデータのバリデーションテスト"""
        result = self.generator.validate_candlestick_data(self.mock_ohlc_data)
        self.assertTrue(result)
    
    def test_validate_candlestick_data_invalid(self):
        """無効なキャンドルスティックデータのバリデーションテスト"""
        # 必要な列が欠けているデータ
        invalid_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            # 'Low'と'Close'が欠けている
        })
        
        result = self.generator.validate_candlestick_data(invalid_data)
        self.assertFalse(result)
    
    def test_validate_candlestick_data_illogical(self):
        """論理的に無効なキャンドルスティックデータのテスト"""
        # 高値が安値より低いデータ
        invalid_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [90, 91],  # 高値が安値より低い
            'Low': [95, 96],
            'Close': [102, 103]
        })
        
        result = self.generator.validate_candlestick_data(invalid_data)
        self.assertTrue(result)  # バリデーションは実行されるが、無効な行は削除される
    
    def test_calculate_moving_averages(self):
        """移動平均計算のテスト"""
        ma_keys = ['ma5', 'ma10']
        result = self.generator._calculate_moving_averages(
            self.mock_ohlc_data, ma_keys, 'SMA'
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        # 移動平均の列が追加されていることを確認
        self.assertIn('MA5', result.columns)
    
    @patch('plotly.io.write_html')
    def test_generate_interactive_chart(self, mock_write_html):
        """インタラクティブチャート生成のテスト"""
        mock_write_html.return_value = None
        
        result = self.generator.generate_interactive_chart(
            self.mock_ohlc_data, 
            "Test Chart", 
            "test_chart.html"
        )
        
        self.assertIsNotNone(result)
        mock_write_html.assert_called_once()
    
    @patch('mplfinance.plot')
    def test_generate_static_chart(self, mock_plot):
        """静的チャート生成のテスト"""
        mock_fig = Mock()
        mock_axlist = [Mock()]
        mock_plot.return_value = (mock_fig, mock_axlist)
        
        with patch.object(self.generator, 'save_chart') as mock_save:
            mock_save.return_value = f"{self.temp_dir}/test_chart.png"
            
            result = self.generator.generate_static_chart(
                self.mock_ohlc_data, 
                "Test Chart", 
                "test_chart.png"
            )
            
            self.assertIsNotNone(result)
            mock_plot.assert_called_once()
            mock_save.assert_called_once()
    
    def test_generate_intraday_chart(self):
        """イントラデイチャート生成のテスト"""
        with patch.object(self.generator, 'generate_interactive_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/test_intraday.html"
            
            result = self.generator.generate_intraday_chart(
                self.mock_ohlc_data, 
                "Test Ticker", 
                "test_intraday.html"
            )
            
            self.assertIsNotNone(result)
            mock_generate.assert_called_once()
    
    def test_generate_longterm_chart(self):
        """長期チャート生成のテスト"""
        with patch.object(self.generator, 'generate_interactive_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/test_longterm.html"
            
            result = self.generator.generate_longterm_chart(
                self.mock_ohlc_data, 
                "Test Ticker", 
                "test_longterm.html"
            )
            
            self.assertIsNotNone(result)
            mock_generate.assert_called_once()
    
    def test_moving_average_config(self):
        """移動平均設定のテスト"""
        # 移動平均設定の取得
        ma_config = self.generator.get_moving_average_config('ma5')
        self.assertIsInstance(ma_config, dict)
        
        # 移動平均設定の追加
        self.generator.add_moving_average('ma20', 20, 'purple', 'MA20')
        
        # 移動平均設定の削除
        self.generator.remove_moving_average('ma20')


class TestSectorChartGenerator(unittest.TestCase):
    """SectorChartGeneratorのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SectorChartGenerator(self.temp_dir, self.logger)
        
        # モックセクターデータ
        self.mock_sector_data = {
            'Technology Select Sector SPDR Fund': 2.5,
            'Financial Select Sector SPDR Fund': 1.8,
            'Health Care Select Sector SPDR Fund': -0.5,
            'Energy Select Sector SPDR Fund': -1.2,
            'Materials Select Sector SPDR Fund': 0.8
        }
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.generator.logger)
        self.assertIsNotNone(self.generator.error_handler)
        self.assertIsNotNone(self.generator.config)
    
    def test_validate_sector_data_valid(self):
        """有効なセクターデータのバリデーションテスト"""
        result = self.generator.validate_sector_data(self.mock_sector_data)
        self.assertTrue(result)
    
    def test_validate_sector_data_invalid(self):
        """無効なセクターデータのバリデーションテスト"""
        invalid_data = {
            'Sector1': 'not_a_number',
            'Sector2': None,
            'Sector3': float('nan')
        }
        
        result = self.generator.validate_sector_data(invalid_data)
        self.assertFalse(result)
    
    def test_prepare_chart_data(self):
        """チャートデータ準備のテスト"""
        result = self.generator._prepare_chart_data(self.mock_sector_data)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # 各データが(名前, 値, 色)のタプルであることを確認
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)
    
    def test_shorten_sector_name(self):
        """セクター名短縮のテスト"""
        long_name = "Technology Select Sector SPDR Fund"
        short_name = self.generator._shorten_sector_name(long_name)
        
        self.assertNotEqual(long_name, short_name)
        self.assertIn("Tech", short_name)
    
    def test_determine_color(self):
        """色決定のテスト"""
        # 正の値
        positive_color = self.generator._determine_color(1.5)
        self.assertIsInstance(positive_color, str)
        
        # 負の値
        negative_color = self.generator._determine_color(-1.5)
        self.assertIsInstance(negative_color, str)
        
        # ゼロ
        neutral_color = self.generator._determine_color(0.0)
        self.assertIsInstance(neutral_color, str)
    
    @patch('plotly.io.write_html')
    def test_generate_interactive_chart(self, mock_write_html):
        """インタラクティブチャート生成のテスト"""
        mock_write_html.return_value = None
        
        result = self.generator.generate_interactive_chart(
            self.mock_sector_data, 
            "Test Sector Chart", 
            "test_sector.html"
        )
        
        self.assertIsNotNone(result)
        mock_write_html.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    def test_generate_static_chart(self, mock_savefig):
        """静的チャート生成のテスト"""
        mock_savefig.return_value = None
        
        with patch.object(self.generator, 'save_chart') as mock_save:
            mock_save.return_value = f"{self.temp_dir}/test_sector.png"
            
            result = self.generator.generate_static_chart(
                self.mock_sector_data, 
                "Test Sector Chart", 
                "test_sector.png"
            )
            
            self.assertIsNotNone(result)
            mock_save.assert_called_once()
    
    def test_generate_sector_performance_chart(self):
        """セクターパフォーマンスチャート生成のテスト"""
        with patch.object(self.generator, 'generate_interactive_chart') as mock_generate:
            mock_generate.return_value = f"{self.temp_dir}/sector_performance.html"
            
            result = self.generator.generate_sector_performance_chart(
                self.mock_sector_data, 
                "sector_performance.html"
            )
            
            self.assertIsNotNone(result)
            mock_generate.assert_called_once()
    
    def test_create_sector_comparison_chart(self):
        """セクター比較チャート生成のテスト"""
        # 複数時期のデータ
        comparison_data = {
            'Period1': self.mock_sector_data,
            'Period2': {k: v + 0.5 for k, v in self.mock_sector_data.items()}
        }
        
        with patch('plotly.io.write_html') as mock_write_html:
            mock_write_html.return_value = None
            
            result = self.generator.create_sector_comparison_chart(
                comparison_data, 
                "Sector Comparison", 
                "comparison.html"
            )
            
            self.assertIsNotNone(result)
            mock_write_html.assert_called_once()
    
    def test_get_sector_statistics(self):
        """セクター統計情報取得のテスト"""
        stats = self.generator.get_sector_statistics(self.mock_sector_data)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('count', stats)
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
    
    def test_color_configuration(self):
        """色設定のテスト"""
        # 色設定の取得
        colors = self.generator.get_sector_colors()
        self.assertIsInstance(colors, dict)
        
        # 色設定の更新
        new_colors = {'positive': 'blue', 'negative': 'orange'}
        self.generator.set_sector_colors(new_colors)


class TestChartGeneratorFactory(unittest.TestCase):
    """ChartGeneratorFactoryのテスト"""
    
    def test_create_candlestick_generator(self):
        """キャンドルスティックジェネレーター作成のテスト"""
        generator = ChartGeneratorFactory.create_generator("candlestick")
        self.assertIsInstance(generator, CandlestickChartGenerator)
    
    def test_create_sector_generator(self):
        """セクタージェネレーター作成のテスト"""
        generator = ChartGeneratorFactory.create_generator("sector")
        self.assertIsInstance(generator, SectorChartGenerator)
    
    def test_create_invalid_generator(self):
        """無効なジェネレーター作成のテスト"""
        with self.assertRaises(ValueError):
            ChartGeneratorFactory.create_generator("invalid_type")
    
    def test_get_available_generators(self):
        """利用可能なジェネレーター取得のテスト"""
        available = ChartGeneratorFactory.get_available_generators()
        self.assertIsInstance(available, list)
        self.assertIn("candlestick", available)
        self.assertIn("sector", available)


class TestChartGeneratorIntegration(unittest.TestCase):
    """チャートジェネレーターの統合テスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.temp_dir = tempfile.mkdtemp()
        self.candlestick_generator = CandlestickChartGenerator(self.temp_dir, self.logger)
        self.sector_generator = SectorChartGenerator(self.temp_dir, self.logger)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_all_generators_initialization(self):
        """すべてのジェネレーターの初期化テスト"""
        generators = [self.candlestick_generator, self.sector_generator]
        
        for generator in generators:
            self.assertIsNotNone(generator.logger)
            self.assertIsNotNone(generator.error_handler)
            self.assertIsNotNone(generator.config)
    
    def test_error_handling_consistency(self):
        """エラーハンドリングの一貫性テスト"""
        test_error = ChartGenerationError("Test error")
        
        # すべてのジェネレーターで同じエラーハンドリングを実行
        self.candlestick_generator._handle_generation_error(test_error, "test")
        self.sector_generator._handle_generation_error(test_error, "test")
        
        # エラーログが出力されることを確認
        self.candlestick_generator.logger.error.assert_called()
        self.sector_generator.logger.error.assert_called()
    
    def test_output_directory_consistency(self):
        """出力ディレクトリの一貫性テスト"""
        new_dir = tempfile.mkdtemp()
        
        generators = [self.candlestick_generator, self.sector_generator]
        
        for generator in generators:
            generator.set_output_directory(new_dir)
            self.assertEqual(str(generator.charts_dir), new_dir)
    
    def test_configuration_access(self):
        """設定アクセスの一貫性テスト"""
        generators = [self.candlestick_generator, self.sector_generator]
        
        for generator in generators:
            self.assertIsNotNone(generator.config)
            # 設定検証を実行
            try:
                is_valid = generator.validate_chart_config()
                self.assertIsInstance(is_valid, bool)
            except Exception as e:
                self.fail(f"Config validation failed for {type(generator).__name__}: {e}")


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)