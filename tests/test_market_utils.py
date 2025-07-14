"""
マーケットユーティリティのテスト
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import pytz

from market_utils import (
    MarketDataProcessor, WeekendHandler, EconomicDataProcessor,
    TextProcessor, ChartDataHelper
)
from config import Config

class MockConfig(Config):
    """テスト用のモック設定"""
    def __init__(self):
        pass
    
    INDICATOR_TRANSLATIONS = {
        "Initial Jobless Claims": "新規失業保険申請件数",
        "GDP": "国内総生産"
    }

class TestMarketDataProcessor:
    """MarketDataProcessor のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        config = MockConfig()
        processor = MarketDataProcessor(config)
        
        assert processor.config == config
        assert processor.logger is not None
    
    def test_process_ticker_data_success(self):
        """ティッカーデータ処理成功のテスト"""
        # テストデータの準備
        test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2025-01-01', periods=5, freq='D'))
        
        config = MockConfig()
        processor = MarketDataProcessor(config)
        
        result = processor.process_ticker_data("TEST", test_data, "US_STOCK")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # 基本的なOHLCデータが保持されていることを確認
        assert 'Open' in result.columns
        assert 'Close' in result.columns
    
    def test_process_ticker_data_empty(self):
        """空のティッカーデータ処理のテスト"""
        config = MockConfig()
        processor = MarketDataProcessor(config)
        
        result = processor.process_ticker_data("TEST", pd.DataFrame(), "US_STOCK")
        assert result is None
        
        result = processor.process_ticker_data("TEST", None, "US_STOCK")
        assert result is None
    
    def test_clean_ohlc_data(self):
        """OHLCデータクリーニングのテスト"""
        # 異常値を含むテストデータ
        test_data = pd.DataFrame({
            'Open': [100, 0, 102, None, 104],  # 0と欠損値を含む
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        config = MockConfig()
        processor = MarketDataProcessor(config)
        
        cleaned_data = processor._clean_ohlc_data(test_data)
        
        # 異常値と欠損値が除去されていることを確認
        assert len(cleaned_data) < len(test_data)
        assert all(cleaned_data['Open'] > 0)
        assert not cleaned_data.isna().any().any()
    
    def test_calculate_rsi(self):
        """RSI計算のテスト"""
        # RSI計算用のテストデータ
        prices = pd.Series([100, 102, 101, 105, 103, 107, 106, 108, 110, 109, 
                           111, 113, 112, 115, 114, 116, 118, 117, 119, 120])
        
        config = MockConfig()
        processor = MarketDataProcessor(config)
        
        rsi = processor._calculate_rsi(prices, 14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        # RSIは0-100の範囲内
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)

class TestWeekendHandler:
    """WeekendHandler のテスト"""
    
    @patch('market_utils.datetime')
    def test_is_weekend_in_timezone_saturday(self, mock_datetime):
        """土曜日の週末判定テスト"""
        # 土曜日をモック
        mock_now = Mock()
        mock_now.weekday.return_value = 5  # 土曜日
        mock_datetime.now.return_value = mock_now
        
        result = WeekendHandler.is_weekend_in_timezone('America/New_York')
        assert result is True
    
    @patch('market_utils.datetime')
    def test_is_weekend_in_timezone_monday(self, mock_datetime):
        """月曜日の週末判定テスト"""
        # 月曜日をモック
        mock_now = Mock()
        mock_now.weekday.return_value = 0  # 月曜日
        mock_datetime.now.return_value = mock_now
        
        result = WeekendHandler.is_weekend_in_timezone('America/New_York')
        assert result is False
    
    def test_get_last_business_day(self):
        """最終営業日取得のテスト"""
        result = WeekendHandler.get_last_business_day()
        
        assert isinstance(result, datetime)
        # 結果が過去または現在の日付であることを確認
        assert result <= datetime.now(pytz.timezone('America/New_York'))

class TestEconomicDataProcessor:
    """EconomicDataProcessor のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        config = MockConfig()
        processor = EconomicDataProcessor(config)
        
        assert processor.config == config
        assert processor.translations == config.INDICATOR_TRANSLATIONS
    
    def test_process_economic_indicators_success(self):
        """経済指標処理成功のテスト"""
        # テストデータの準備
        test_data = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'event': ['GDP', 'Initial Jobless Claims', 'Unemployment Rate'],
            'actual': ['2.1%', '350K', '3.5%'],
            'forecast': ['2.0%', '360K', '3.6%'],
            'previous': ['1.9%', '340K', '3.7%'],
            'time': ['10:00', '08:30', '08:30']
        })
        
        config = MockConfig()
        processor = EconomicDataProcessor(config)
        
        with patch('market_utils.datetime') as mock_datetime:
            # 現在日時を2025-01-02に設定
            mock_datetime.now.return_value.date.return_value = datetime(2025, 1, 2).date()
            
            result = processor.process_economic_indicators(test_data)
        
        assert isinstance(result, dict)
        assert 'yesterday' in result
        assert 'today_scheduled' in result
        assert isinstance(result['yesterday'], list)
        assert isinstance(result['today_scheduled'], list)
    
    def test_process_economic_indicators_empty(self):
        """空の経済指標処理のテスト"""
        config = MockConfig()
        processor = EconomicDataProcessor(config)
        
        result = processor.process_economic_indicators(pd.DataFrame())
        
        assert result == {"yesterday": [], "today_scheduled": []}
        
        result = processor.process_economic_indicators(None)
        assert result == {"yesterday": [], "today_scheduled": []}
    
    def test_translate_indicator_name(self):
        """経済指標名翻訳のテスト"""
        config = MockConfig()
        processor = EconomicDataProcessor(config)
        
        # 翻訳辞書にある場合
        translated = processor._translate_indicator_name("Initial Jobless Claims")
        assert translated == "新規失業保険申請件数"
        
        # 翻訳辞書にない場合（元の名前を返す）
        not_translated = processor._translate_indicator_name("Unknown Indicator")
        assert not_translated == "Unknown Indicator"

class TestTextProcessor:
    """TextProcessor のテスト"""
    
    def test_clean_text_for_ai_basic(self):
        """基本的なテキストクリーニングのテスト"""
        dirty_text = "<p>This is a   test  text\r\nwith HTML tags</p>"
        
        cleaned = TextProcessor.clean_text_for_ai(dirty_text)
        
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "This is a test text\nwith HTML tags" == cleaned
    
    def test_clean_text_for_ai_empty(self):
        """空テキストのクリーニングテスト"""
        assert TextProcessor.clean_text_for_ai("") == ""
        assert TextProcessor.clean_text_for_ai(None) == ""
    
    def test_clean_text_for_ai_length_limit(self):
        """長さ制限のテスト"""
        long_text = "A" * 100
        
        cleaned = TextProcessor.clean_text_for_ai(long_text, max_length=50)
        
        assert len(cleaned) <= 53  # 50 + "..." = 53
        assert cleaned.endswith("...")
    
    def test_extract_keywords(self):
        """キーワード抽出のテスト"""
        text = "The stock market is performing well today with good earnings reports"
        
        keywords = TextProcessor.extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # 長さ3以上の単語のみ含まれることを確認
        assert all(len(word) >= 3 for word in keywords)
        # 重複がないことを確認
        assert len(keywords) == len(set(keywords))
    
    def test_extract_keywords_empty(self):
        """空テキストのキーワード抽出テスト"""
        assert TextProcessor.extract_keywords("") == []
        assert TextProcessor.extract_keywords(None) == []

class TestChartDataHelper:
    """ChartDataHelper のテスト"""
    
    def test_prepare_chart_metadata(self):
        """チャートメタデータ準備のテスト"""
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2025-01-01', periods=3, freq='D'))
        
        metadata = ChartDataHelper.prepare_chart_metadata("TEST", test_data, "longterm")
        
        assert isinstance(metadata, dict)
        assert metadata['ticker'] == "TEST"
        assert metadata['chart_type'] == "longterm"
        assert metadata['data_points'] == 3
        assert 'date_range' in metadata
        assert 'price_range' in metadata
        assert metadata['price_range']['min'] == 95.0
        assert metadata['price_range']['max'] == 107.0
    
    def test_prepare_chart_metadata_empty(self):
        """空データのチャートメタデータ準備テスト"""
        metadata = ChartDataHelper.prepare_chart_metadata("TEST", pd.DataFrame(), "longterm")
        
        assert metadata == {}
        
        metadata = ChartDataHelper.prepare_chart_metadata("TEST", None, "longterm")
        assert metadata == {}
    
    def test_add_cache_buster(self):
        """キャッシュバスター追加のテスト"""
        original_path = "/path/to/chart.png"
        
        busted_path = ChartDataHelper.add_cache_buster(original_path)
        
        assert busted_path.startswith(original_path)
        assert "?v=" in busted_path
        # タイムスタンプが数値であることを確認
        timestamp = busted_path.split("?v=")[1]
        assert timestamp.isdigit()