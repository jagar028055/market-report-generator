"""
ユーティリティモジュールのテスト
"""

import pytest
import pandas as pd
import time
from unittest.mock import patch, MagicMock
import requests

from utils import (
    retry_on_error, retry_api_call, retry_network_operation,
    validate_dataframe, validate_market_data, validate_news_data,
    safe_request, DataValidator
)

class TestRetryDecorators:
    """リトライデコレータのテスト"""
    
    def test_retry_on_error_success(self):
        """成功時のリトライテスト"""
        call_count = 0
        
        @retry_on_error(max_attempts=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_error_failure_then_success(self):
        """失敗後成功のリトライテスト"""
        call_count = 0
        
        @retry_on_error(max_attempts=3, wait_min=0, wait_max=0)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_error_max_attempts_exceeded(self):
        """最大試行回数超過のテスト"""
        call_count = 0
        
        @retry_on_error(max_attempts=2, wait_min=0, wait_max=0)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            test_function()
        assert call_count == 2
    
    def test_retry_api_call(self):
        """API呼び出しリトライのテスト"""
        @retry_api_call(max_attempts=2)
        def mock_api_call():
            return {"status": "ok"}
        
        result = mock_api_call()
        assert result["status"] == "ok"

class TestDataValidation:
    """データ検証のテスト"""
    
    def test_validate_dataframe_valid(self):
        """有効なDataFrameの検証"""
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        assert validate_dataframe(df, ['Open', 'High', 'Low', 'Close'], min_rows=3)
        assert validate_dataframe(df, min_rows=1)
    
    def test_validate_dataframe_empty(self):
        """空のDataFrameの検証"""
        df = pd.DataFrame()
        
        assert not validate_dataframe(df)
        assert not validate_dataframe(None)
    
    def test_validate_dataframe_missing_columns(self):
        """必須カラム不足のDataFrame検証"""
        df = pd.DataFrame({
            'Open': [100, 101],
            'Close': [103, 104]
        })
        
        assert not validate_dataframe(df, ['Open', 'High', 'Low', 'Close'])
        assert validate_dataframe(df, ['Open', 'Close'])
    
    def test_validate_market_data_valid(self):
        """有効なマーケットデータの検証"""
        market_data = {
            "S&P500": {
                "current": "4500.00",
                "change": "+10.50",
                "change_percent": "+0.23%"
            },
            "NASDAQ": {
                "current": "15000.00",
                "change": "-5.25",
                "change_percent": "-0.03%"
            }
        }
        
        assert validate_market_data(market_data)
    
    def test_validate_market_data_invalid(self):
        """無効なマーケットデータの検証"""
        # 空のデータ
        assert not validate_market_data({})
        
        # 必須フィールド不足
        invalid_data = {
            "S&P500": {
                "current": "4500.00"
                # change, change_percent が不足
            }
        }
        assert not validate_market_data(invalid_data)
        
        # 無効なデータ型
        invalid_type_data = {
            "S&P500": "not a dict"
        }
        assert not validate_market_data(invalid_type_data)
    
    def test_validate_news_data_valid(self):
        """有効なニュースデータの検証"""
        news_data = [
            {
                "title": "Market Update Today",
                "url": "https://example.com/news1",
                "published_jst": "2025-01-01 10:00:00",
                "country": "US"
            },
            {
                "title": "Economic Report Released",
                "url": "https://example.com/news2",
                "published_jst": "2025-01-01 11:00:00",
                "country": "US"
            }
        ]
        
        assert validate_news_data(news_data)
    
    def test_validate_news_data_invalid(self):
        """無効なニュースデータの検証"""
        # 無効なURL
        invalid_news = [
            {
                "title": "Test News",
                "url": "invalid-url",
                "published_jst": "2025-01-01 10:00:00",
                "country": "US"
            }
        ]
        assert not validate_news_data(invalid_news)
        
        # 必須フィールド不足
        missing_field_news = [
            {
                "title": "Test News",
                # url が不足
                "published_jst": "2025-01-01 10:00:00",
                "country": "US"
            }
        ]
        assert not validate_news_data(missing_field_news)
    
    def test_validate_news_data_empty(self):
        """空のニュースデータの検証"""
        assert validate_news_data([])  # 空は有効とする

class TestDataValidator:
    """DataValidatorクラスのテスト"""
    
    def test_validate_ticker_data_valid(self):
        """有効なティッカーデータの検証"""
        ticker_data = {
            "current": "100.50",
            "change": "+1.25",
            "change_percent": "+1.26%"
        }
        
        assert DataValidator.validate_ticker_data("TEST", ticker_data)
    
    def test_validate_ticker_data_with_na_values(self):
        """N/A値を含むティッカーデータの検証"""
        ticker_data = {
            "current": "100.50",
            "change": "N/A",
            "change_percent": "N/A"
        }
        
        assert DataValidator.validate_ticker_data("TEST", ticker_data)
    
    def test_validate_ticker_data_invalid_numeric(self):
        """無効な数値を含むティッカーデータの検証"""
        ticker_data = {
            "current": "invalid_number",
            "change": "+1.25",
            "change_percent": "+1.26%"
        }
        
        assert not DataValidator.validate_ticker_data("TEST", ticker_data)
    
    def test_validate_chart_data_valid(self):
        """有効なチャートデータの検証"""
        chart_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        assert DataValidator.validate_chart_data(chart_data, "TEST")
    
    def test_validate_chart_data_invalid(self):
        """無効なチャートデータの検証"""
        # 必須カラム不足
        invalid_data = pd.DataFrame({'Close': [100, 101]})
        assert not DataValidator.validate_chart_data(invalid_data, "TEST")
        
        # 空のデータ
        empty_data = pd.DataFrame()
        assert not DataValidator.validate_chart_data(empty_data, "TEST")

class TestSafeRequest:
    """safe_request関数のテスト"""
    
    @patch('utils.requests.get')
    def test_safe_request_success(self, mock_get):
        """成功時のsafe_requestテスト"""
        # モックレスポンスの設定
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = safe_request("https://example.com")
        assert result == mock_response
        mock_get.assert_called_once()
    
    @patch('utils.requests.get')
    def test_safe_request_with_retry(self, mock_get):
        """リトライ機能のテスト"""
        # 最初は失敗、2回目は成功
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response
        ]
        
        result = safe_request("https://example.com", max_retries=2)
        assert result == mock_response
        assert mock_get.call_count == 2

class TestPerformanceMetrics:
    """パフォーマンスメトリクスのテスト"""
    
    def test_measure_time_decorator(self):
        """実行時間計測デコレータのテスト"""
        from utils import measure_time
        
        execution_times = []
        
        @measure_time
        def test_function():
            time.sleep(0.1)  # 100ms待機
            return "done"
        
        result = test_function()
        assert result == "done"
        # 実際の計測時間確認は複雑なので、正常実行のみ確認