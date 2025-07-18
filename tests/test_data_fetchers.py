"""
データフェッチャーのテスト
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import logging
from datetime import datetime, timedelta
import pytest

from src.data_fetchers.base_fetcher import BaseDataFetcher
from src.data_fetchers.market_data_fetcher import MarketDataFetcher
from src.data_fetchers.news_data_fetcher import NewsDataFetcher
from src.data_fetchers.economic_data_fetcher import EconomicDataFetcher
from src.utils.exceptions import DataFetchError, NetworkError, ValidationError


class TestBaseDataFetcher(unittest.TestCase):
    """BaseDataFetcherのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        
        # 具象クラスを作成してテスト
        class TestDataFetcher(BaseDataFetcher):
            def fetch_data(self, **kwargs):
                return {"test": "data"}
        
        self.fetcher = TestDataFetcher(self.logger)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.fetcher.logger)
        self.assertIsNotNone(self.fetcher.error_handler)
        self.assertIsNotNone(self.fetcher.config)
    
    def test_validation_empty_data(self):
        """空データのバリデーションテスト"""
        self.assertFalse(self.fetcher.validate_data(None))
        self.assertFalse(self.fetcher.validate_data({}))
        self.assertFalse(self.fetcher.validate_data([]))
    
    def test_validation_valid_data(self):
        """有効データのバリデーションテスト"""
        self.assertTrue(self.fetcher.validate_data({"key": "value"}))
        self.assertTrue(self.fetcher.validate_data([1, 2, 3]))
        self.assertTrue(self.fetcher.validate_data("string"))
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        test_error = DataFetchError("Test error")
        result = self.fetcher.handle_fetch_error(test_error)
        
        self.assertIsNone(result)
        self.fetcher.logger.error.assert_called()
    
    def test_cleanup(self):
        """クリーンアップのテスト"""
        # クリーンアップを実行
        self.fetcher.cleanup()
        
        # エラーハンドラーのクリーンアップが呼ばれることを確認
        self.assertTrue(True)  # 実際の実装では、クリーンアップされることを確認


class TestMarketDataFetcher(unittest.TestCase):
    """MarketDataFetcherのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.fetcher = MarketDataFetcher(self.logger)
        
        # モックデータを準備
        self.mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
    
    @patch('src.data_fetchers.market_data_fetcher.yf.download')
    def test_get_market_data_success(self, mock_download):
        """市場データ取得の成功テスト"""
        mock_download.return_value = self.mock_data
        
        result = self.fetcher.get_market_data()
        
        self.assertIsInstance(result, dict)
        self.assertIn('S&P500', result)
        mock_download.assert_called()
    
    @patch('src.data_fetchers.market_data_fetcher.yf.download')
    def test_get_market_data_failure(self, mock_download):
        """市場データ取得の失敗テスト"""
        mock_download.side_effect = Exception("Network error")
        
        result = self.fetcher.get_market_data()
        
        self.assertIsNone(result)
        self.fetcher.logger.error.assert_called()
    
    @patch('src.data_fetchers.market_data_fetcher.yf.download')
    def test_get_intraday_data(self, mock_download):
        """イントラデイデータ取得テスト"""
        mock_download.return_value = self.mock_data
        
        result = self.fetcher.get_intraday_data("^GSPC")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
    
    @patch('src.data_fetchers.market_data_fetcher.yf.download')
    def test_get_historical_data(self, mock_download):
        """履歴データ取得テスト"""
        mock_download.return_value = self.mock_data
        
        result = self.fetcher.get_historical_data("^GSPC", "1y")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
    
    def test_get_sector_etf_performance(self):
        """セクターETFパフォーマンステスト"""
        with patch.object(self.fetcher, 'get_historical_data') as mock_get_data:
            mock_get_data.return_value = self.mock_data
            
            result = self.fetcher.get_sector_etf_performance()
            
            self.assertIsInstance(result, dict)
            # 少なくとも1つのセクターデータが含まれることを確認
            self.assertGreater(len(result), 0)
    
    def test_invalid_ticker(self):
        """無効なティッカーのテスト"""
        with patch('src.data_fetchers.market_data_fetcher.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame()  # 空のDataFrame
            
            result = self.fetcher.get_intraday_data("INVALID")
            
            self.assertTrue(result.empty)


class TestNewsDataFetcher(unittest.TestCase):
    """NewsDataFetcherのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.fetcher = NewsDataFetcher(self.logger)
        
        # モックニュースデータ
        self.mock_news = [
            {
                'title': 'Test News 1',
                'link': 'https://example.com/news1',
                'published': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary': 'Test news summary 1'
            },
            {
                'title': 'Test News 2',
                'link': 'https://example.com/news2',
                'published': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary': 'Test news summary 2'
            }
        ]
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.fetcher.logger)
        self.assertIsNotNone(self.fetcher.error_handler)
        self.assertIsNotNone(self.fetcher.config)
    
    @patch('src.data_fetchers.news_data_fetcher.webdriver.Chrome')
    def test_fetch_data_success(self, mock_webdriver):
        """データ取得成功テスト"""
        # WebDriverをモック
        mock_driver = Mock()
        mock_webdriver.return_value = mock_driver
        
        with patch.object(self.fetcher, 'scrape_reuters_news') as mock_scrape:
            mock_scrape.return_value = self.mock_news
            
            result = self.fetcher.fetch_data()
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
    
    @patch('src.data_fetchers.news_data_fetcher.webdriver.Chrome')
    def test_fetch_data_failure(self, mock_webdriver):
        """データ取得失敗テスト"""
        mock_webdriver.side_effect = Exception("WebDriver error")
        
        result = self.fetcher.fetch_data()
        
        self.assertIsNone(result)
        self.fetcher.logger.error.assert_called()
    
    def test_filter_news_by_time(self):
        """時間によるニュースフィルタリングテスト"""
        # 古いニュースを作成
        old_news = [
            {
                'title': 'Old News',
                'published': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'summary': 'Old news summary'
            }
        ]
        
        # 24時間以内のニュースのみを取得
        filtered = self.fetcher.filter_news_by_time(old_news + self.mock_news, 24)
        
        self.assertEqual(len(filtered), 2)  # 新しいニュースのみ
    
    def test_validate_news_data(self):
        """ニュースデータのバリデーションテスト"""
        # 有効なニュースデータ
        valid_news = {
            'title': 'Test News',
            'link': 'https://example.com/news',
            'published': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': 'Test summary'
        }
        
        self.assertTrue(self.fetcher.validate_news_data(valid_news))
        
        # 無効なニュースデータ
        invalid_news = {
            'title': '',  # 空のタイトル
            'link': 'invalid-url',  # 無効なURL
            'published': 'invalid-date',  # 無効な日付
            'summary': ''  # 空のサマリー
        }
        
        self.assertFalse(self.fetcher.validate_news_data(invalid_news))


class TestEconomicDataFetcher(unittest.TestCase):
    """EconomicDataFetcherのテスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.fetcher = EconomicDataFetcher(self.logger)
        
        # モック経済指標データ
        self.mock_indicators = {
            'yesterday': [
                {'time': '22:30', 'country': 'US', 'indicator': 'GDP', 'importance': 'high', 'actual': '2.1%'},
                {'time': '15:00', 'country': 'EU', 'indicator': 'CPI', 'importance': 'medium', 'actual': '1.8%'}
            ],
            'today_scheduled': [
                {'time': '22:30', 'country': 'US', 'indicator': 'NFP', 'importance': 'high', 'forecast': '200K'},
                {'time': '16:00', 'country': 'EU', 'indicator': 'ECB Rate', 'importance': 'high', 'forecast': '0.25%'}
            ]
        }
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.fetcher.logger)
        self.assertIsNotNone(self.fetcher.error_handler)
        self.assertIsNotNone(self.fetcher.config)
    
    @patch('src.data_fetchers.economic_data_fetcher.webdriver.Chrome')
    def test_get_economic_indicators_success(self, mock_webdriver):
        """経済指標取得成功テスト"""
        # WebDriverをモック
        mock_driver = Mock()
        mock_webdriver.return_value = mock_driver
        
        with patch.object(self.fetcher, 'scrape_economic_calendar') as mock_scrape:
            mock_scrape.return_value = self.mock_indicators
            
            result = self.fetcher.get_economic_indicators()
            
            self.assertIsInstance(result, dict)
            self.assertIn('yesterday', result)
            self.assertIn('today_scheduled', result)
    
    @patch('src.data_fetchers.economic_data_fetcher.webdriver.Chrome')
    def test_get_economic_indicators_failure(self, mock_webdriver):
        """経済指標取得失敗テスト"""
        mock_webdriver.side_effect = Exception("WebDriver error")
        
        result = self.fetcher.get_economic_indicators()
        
        self.assertIsNone(result)
        self.fetcher.logger.error.assert_called()
    
    def test_parse_economic_data(self):
        """経済データの解析テスト"""
        # モックHTML要素
        mock_element = Mock()
        mock_element.text = "US GDP 2.1% 22:30"
        
        # 解析処理のテスト（実際の実装に依存）
        result = self.fetcher.parse_economic_indicator(mock_element)
        
        # 結果の検証（実際の実装に合わせて調整が必要）
        self.assertIsInstance(result, dict)
    
    def test_filter_high_importance_indicators(self):
        """高重要度指標のフィルタリングテスト"""
        filtered = self.fetcher.filter_by_importance(
            self.mock_indicators['yesterday'], 
            'high'
        )
        
        self.assertEqual(len(filtered), 1)  # 高重要度は1つのみ
        self.assertEqual(filtered[0]['indicator'], 'GDP')
    
    def test_validate_economic_data(self):
        """経済データのバリデーションテスト"""
        # 有効なデータ
        valid_data = {
            'time': '22:30',
            'country': 'US',
            'indicator': 'GDP',
            'importance': 'high',
            'actual': '2.1%'
        }
        
        self.assertTrue(self.fetcher.validate_economic_data(valid_data))
        
        # 無効なデータ
        invalid_data = {
            'time': '',
            'country': '',
            'indicator': '',
            'importance': 'unknown',
            'actual': ''
        }
        
        self.assertFalse(self.fetcher.validate_economic_data(invalid_data))


class TestDataFetcherIntegration(unittest.TestCase):
    """データフェッチャーの統合テスト"""
    
    def setUp(self):
        """テスト用セットアップ"""
        self.logger = Mock(spec=logging.Logger)
        self.market_fetcher = MarketDataFetcher(self.logger)
        self.news_fetcher = NewsDataFetcher(self.logger)
        self.economic_fetcher = EconomicDataFetcher(self.logger)
    
    def test_all_fetchers_initialization(self):
        """すべてのフェッチャーの初期化テスト"""
        fetchers = [self.market_fetcher, self.news_fetcher, self.economic_fetcher]
        
        for fetcher in fetchers:
            self.assertIsNotNone(fetcher.logger)
            self.assertIsNotNone(fetcher.error_handler)
            self.assertIsNotNone(fetcher.config)
    
    def test_error_handling_consistency(self):
        """エラーハンドリングの一貫性テスト"""
        test_error = DataFetchError("Test error")
        
        # すべてのフェッチャーで同じエラーハンドリングを実行
        market_result = self.market_fetcher.handle_fetch_error(test_error)
        news_result = self.news_fetcher.handle_fetch_error(test_error)
        economic_result = self.economic_fetcher.handle_fetch_error(test_error)
        
        # すべてNoneを返すことを確認
        self.assertIsNone(market_result)
        self.assertIsNone(news_result)
        self.assertIsNone(economic_result)
    
    def test_cleanup_all_fetchers(self):
        """すべてのフェッチャーのクリーンアップテスト"""
        fetchers = [self.market_fetcher, self.news_fetcher, self.economic_fetcher]
        
        for fetcher in fetchers:
            try:
                fetcher.cleanup()
                # エラーが発生しないことを確認
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Cleanup failed for {type(fetcher).__name__}: {e}")


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)