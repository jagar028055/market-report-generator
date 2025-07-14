"""
統合テスト
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime

from async_data_fetcher import AsyncDataFetcher, AsyncNewsAggregator, AsyncCommentaryGenerator
from api_clients import APIClientFactory
from config import Config
import market_utils

class MockConfig(Config):
    """テスト用のモック設定"""
    def __init__(self):
        pass
    
    # テスト用の設定値
    MARKET_TICKERS = {
        "S&P500": "^GSPC",
        "NASDAQ": "^NDX"
    }
    SECTOR_ETFS = {
        "XLK": "Technology Select Sector SPDR Fund"
    }
    ASSET_CLASSES = {
        "US_STOCK": ["^GSPC", "^NDX"],
        "24H_ASSET": ["BTC-USD"]
    }
    INTRADAY_PERIOD_DAYS = 7
    INTRADAY_INTERVAL = "5m"
    CHART_LONGTERM_PERIOD = "1y"
    TARGET_CALENDAR_COUNTRIES = ['united states']
    MAX_WORKERS = 2
    AI_TEXT_LIMIT = 1000
    REUTERS_SEARCH_QUERY = "test query"
    REUTERS_MAX_PAGES = 2

class TestAsyncDataFetcher:
    """AsyncDataFetcher の統合テスト"""
    
    @pytest.mark.asyncio
    async def test_fetch_all_market_data_success(self):
        """マーケットデータ全体取得の成功テスト"""
        config = MockConfig()
        fetcher = AsyncDataFetcher(config)
        
        # モックデータを準備
        mock_ticker_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        mock_economic_data = pd.DataFrame({
            'date': ['2025-01-01'],
            'event': ['GDP'],
            'actual': ['2.1%']
        })
        
        # APIクライアントをモック
        with patch.object(APIClientFactory, 'create_yfinance_client') as mock_yf_factory, \
             patch.object(APIClientFactory, 'create_investpy_client') as mock_inv_factory:
            
            # YFinanceクライアントのモック
            mock_yf_client = Mock()
            mock_yf_client.fetch_ticker_data.return_value = mock_ticker_data
            mock_yf_factory.return_value = mock_yf_client
            
            # Investpyクライアントのモック
            mock_inv_client = Mock()
            mock_inv_client.fetch_economic_calendar.return_value = mock_economic_data
            mock_inv_factory.return_value = mock_inv_client
            
            # MarketDataProcessorをモック
            with patch.object(market_utils.MarketDataProcessor, 'process_ticker_data') as mock_processor:
                mock_processor.return_value = mock_ticker_data
                
                result = await fetcher.fetch_all_market_data()
        
        assert isinstance(result, dict)
        assert 'market_data' in result
        assert 'intraday_data' in result
        assert 'longterm_data' in result
        assert 'sector_data' in result
        assert 'economic_indicators' in result
        assert 'execution_summary' in result
        
        # 実行サマリーの確認
        summary = result['execution_summary']
        assert summary['total_tasks'] > 0
        assert summary['successful_tasks'] >= 0
        assert summary['failed_tasks'] >= 0
    
    @pytest.mark.asyncio
    async def test_fetch_ticker_data_async_success(self):
        """個別ティッカーデータ取得の非同期テスト"""
        config = MockConfig()
        fetcher = AsyncDataFetcher(config)
        
        mock_data = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [95], 
            'Close': [103], 'Volume': [1000]
        })
        
        with patch.object(fetcher, '_fetch_ticker_sync') as mock_sync_fetch:
            mock_sync_fetch.return_value = mock_data
            
            result = await fetcher._fetch_ticker_data_async(
                "test_task", "^GSPC", "1d", "1m", "US_STOCK"
            )
        
        assert result.success is True
        assert result.task_id == "test_task"
        assert result.data is not None
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_fetch_ticker_data_async_failure(self):
        """ティッカーデータ取得失敗の非同期テスト"""
        config = MockConfig()
        fetcher = AsyncDataFetcher(config)
        
        with patch.object(fetcher, '_fetch_ticker_sync') as mock_sync_fetch:
            mock_sync_fetch.side_effect = Exception("API Error")
            
            result = await fetcher._fetch_ticker_data_async(
                "test_task", "^GSPC", "1d", "1m", "US_STOCK"
            )
        
        assert result.success is False
        assert result.task_id == "test_task"
        assert result.data is None
        assert result.error == "API Error"

class TestAsyncNewsAggregator:
    """AsyncNewsAggregator の統合テスト"""
    
    @pytest.mark.asyncio
    async def test_fetch_all_news_success(self):
        """ニュース取得成功の非同期テスト"""
        config = MockConfig()
        aggregator = AsyncNewsAggregator(config)
        
        mock_articles = [
            {
                'title': 'Market Update',
                'url': 'https://example.com/news1',
                'published_date': '2025-01-01',
                'country': 'US'
            },
            {
                'title': 'Economic Report',
                'url': 'https://example.com/news2',
                'published_date': '2025-01-01',
                'country': 'US'
            }
        ]
        
        with patch.object(aggregator, '_fetch_reuters_news_sync') as mock_sync_fetch:
            mock_sync_fetch.return_value = mock_articles
            
            result = await aggregator.fetch_all_news()
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all('title' in article for article in result)
        assert all('url' in article for article in result)
    
    @pytest.mark.asyncio
    async def test_fetch_all_news_failure(self):
        """ニュース取得失敗の非同期テスト"""
        config = MockConfig()
        aggregator = AsyncNewsAggregator(config)
        
        with patch.object(aggregator, '_fetch_reuters_news_sync') as mock_sync_fetch:
            mock_sync_fetch.side_effect = Exception("Network Error")
            
            result = await aggregator.fetch_all_news()
        
        assert isinstance(result, list)
        assert len(result) == 0

class TestAsyncCommentaryGenerator:
    """AsyncCommentaryGenerator の統合テスト"""
    
    @pytest.mark.asyncio
    async def test_generate_commentary_success(self):
        """コメント生成成功の非同期テスト"""
        config = MockConfig()
        generator = AsyncCommentaryGenerator(config)
        
        mock_market_data = {
            "S&P500": {"current": "4500.00", "change": "+10.50"}
        }
        mock_news_articles = [
            {"title": "Market Rally Continues", "url": "https://example.com/news1"}
        ]
        
        with patch.object(generator, '_generate_commentary_sync') as mock_sync_generate:
            mock_sync_generate.return_value = "Generated market commentary"
            
            result = await generator.generate_commentary(mock_market_data, mock_news_articles)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "Generated market commentary"
    
    @pytest.mark.asyncio
    async def test_generate_commentary_failure(self):
        """コメント生成失敗の非同期テスト"""
        config = MockConfig()
        generator = AsyncCommentaryGenerator(config)
        
        mock_market_data = {}
        mock_news_articles = []
        
        with patch.object(generator, '_generate_commentary_sync') as mock_sync_generate:
            mock_sync_generate.side_effect = Exception("AI Service Error")
            
            result = await generator.generate_commentary(mock_market_data, mock_news_articles)
        
        assert isinstance(result, str)
        assert "エラー" in result

class TestFullPipelineIntegration:
    """フルパイプライン統合テスト"""
    
    @pytest.mark.asyncio
    async def test_main_async_execution_success(self):
        """メイン非同期実行の成功テスト"""
        from async_data_fetcher import main_async_execution
        
        # 必要なモックを設定
        mock_market_data = {
            "intraday_data": {"S&P500": pd.DataFrame()},
            "longterm_data": {"S&P500": pd.DataFrame()},
            "execution_summary": {"total_tasks": 1, "successful_tasks": 1}
        }
        mock_news_articles = [
            {"title": "Test News", "url": "https://example.com/test"}
        ]
        mock_commentary = "Generated test commentary"
        
        with patch('async_data_fetcher.AsyncDataFetcher') as mock_fetcher_class, \
             patch('async_data_fetcher.AsyncNewsAggregator') as mock_news_class, \
             patch('async_data_fetcher.AsyncCommentaryGenerator') as mock_commentary_class:
            
            # モックインスタンスの設定
            mock_fetcher = Mock()
            mock_fetcher.fetch_all_market_data.return_value = mock_market_data
            mock_fetcher_class.return_value = mock_fetcher
            
            mock_news_aggregator = Mock()
            mock_news_aggregator.fetch_all_news.return_value = mock_news_articles
            mock_news_class.return_value = mock_news_aggregator
            
            mock_commentary_generator = Mock()
            mock_commentary_generator.generate_commentary.return_value = mock_commentary
            mock_commentary_class.return_value = mock_commentary_generator
            
            result = await main_async_execution()
        
        assert isinstance(result, dict)
        assert 'market_data' in result
        assert 'news_articles' in result
        assert 'commentary' in result
        
        assert result['market_data'] == mock_market_data
        assert result['news_articles'] == mock_news_articles
        assert result['commentary'] == mock_commentary
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self):
        """同時データ取得のパフォーマンステスト"""
        config = MockConfig()
        
        # 複数のタスクを同時実行
        tasks = []
        
        # データ取得タスク
        fetcher = AsyncDataFetcher(config)
        tasks.append(fetcher.fetch_all_market_data())
        
        # ニュース取得タスク
        news_aggregator = AsyncNewsAggregator(config)
        tasks.append(news_aggregator.fetch_all_news())
        
        # モックを設定して実行
        with patch.object(APIClientFactory, 'create_yfinance_client'), \
             patch.object(APIClientFactory, 'create_investpy_client'), \
             patch.object(APIClientFactory, 'create_reuters_client'), \
             patch.object(market_utils.MarketDataProcessor, 'process_ticker_data'):
            
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = asyncio.get_event_loop().time()
            
            execution_time = end_time - start_time
        
        # 結果の確認
        assert len(results) == 2
        # パフォーマンス確認（実際の実装では調整が必要）
        assert execution_time < 10.0  # 10秒以内に完了することを期待