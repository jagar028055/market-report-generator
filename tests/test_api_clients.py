"""
APIクライアントのテスト
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from api_clients import (
    YFinanceClient, InvestpyClient, GeminiClient, 
    APIClientFactory, BaseAPIClient
)
from config import Config

class MockConfig(Config):
    """テスト用のモック設定"""
    def __init__(self):
        # 親クラスの初期化をスキップ
        pass
    
    # テスト用の設定値
    MARKET_TICKERS = {"TEST": "^TEST"}
    GEMINI_PREFERRED_MODELS = ["test-model"]
    AI_TEXT_LIMIT = 1000
    TIMEOUT_SECONDS = 30
    RETRY_ATTEMPTS = 3

class TestBaseAPIClient:
    """BaseAPIClient のテスト"""
    
    def test_base_api_client_initialization(self):
        """基底クラスの初期化テスト"""
        # 抽象クラスなので直接インスタンス化はできないが、継承クラスで確認
        config = MockConfig()
        client = YFinanceClient(config)
        
        assert client.config == config
        assert client.logger is not None
        assert client.metrics_logger is not None

class TestYFinanceClient:
    """YFinanceClient のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        config = MockConfig()
        client = YFinanceClient(config)
        
        assert isinstance(client, YFinanceClient)
        assert client.config == config
    
    @patch('api_clients.yf.Ticker')
    def test_fetch_ticker_data_success(self, mock_ticker_class):
        """ティッカーデータ取得成功のテスト"""
        # モックデータの準備
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        config = MockConfig()
        client = YFinanceClient(config)
        
        result = client.fetch_ticker_data("TEST", "1d", "1m")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        mock_ticker_class.assert_called_once_with("TEST")
        mock_ticker.history.assert_called_once_with(period="1d", interval="1m")
    
    @patch('api_clients.yf.Ticker')
    def test_fetch_ticker_data_empty_result(self, mock_ticker_class):
        """空のティッカーデータのテスト"""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        config = MockConfig()
        client = YFinanceClient(config)
        
        result = client.fetch_ticker_data("TEST", "1d", "1m")
        
        assert result is None
    
    @patch('api_clients.yf.Ticker')
    def test_fetch_ticker_info_success(self, mock_ticker_class):
        """ティッカー情報取得成功のテスト"""
        mock_info = {
            'regularMarketPrice': 100.50,
            'marketCap': 1000000000,
            'shortName': 'Test Company'
        }
        
        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker
        
        config = MockConfig()
        client = YFinanceClient(config)
        
        result = client.fetch_ticker_info("TEST")
        
        assert result == mock_info
        assert 'regularMarketPrice' in result
    
    @patch('api_clients.yf.Ticker')
    def test_fetch_ticker_info_no_price(self, mock_ticker_class):
        """価格情報なしのティッカー情報テスト"""
        mock_info = {'shortName': 'Test Company'}  # regularMarketPrice がない
        
        mock_ticker = Mock()
        mock_ticker.info = mock_info
        mock_ticker_class.return_value = mock_ticker
        
        config = MockConfig()
        client = YFinanceClient(config)
        
        result = client.fetch_ticker_info("TEST")
        
        assert result is None

class TestInvestpyClient:
    """InvestpyClient のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        config = MockConfig()
        client = InvestpyClient(config)
        
        assert isinstance(client, InvestpyClient)
    
    @patch('api_clients.investpy.economic_calendar')
    def test_fetch_economic_calendar_success(self, mock_economic_calendar):
        """経済カレンダー取得成功のテスト"""
        mock_data = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'event': ['GDP', 'Unemployment Rate'],
            'actual': ['2.1%', '3.5%'],
            'forecast': ['2.0%', '3.6%'],
            'previous': ['1.9%', '3.7%']
        })
        
        mock_economic_calendar.return_value = mock_data
        
        config = MockConfig()
        client = InvestpyClient(config)
        
        result = client.fetch_economic_calendar(
            countries=['united states'],
            from_date='2025-01-01',
            to_date='2025-01-02'
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_economic_calendar.assert_called_once()
    
    @patch('api_clients.investpy.economic_calendar')
    def test_fetch_economic_calendar_empty(self, mock_economic_calendar):
        """空の経済カレンダーのテスト"""
        mock_economic_calendar.return_value = pd.DataFrame()
        
        config = MockConfig()
        client = InvestpyClient(config)
        
        result = client.fetch_economic_calendar(
            countries=['united states'],
            from_date='2025-01-01',
            to_date='2025-01-02'
        )
        
        assert result is None

class TestGeminiClient:
    """GeminiClient のテスト"""
    
    @patch('api_clients.genai.GenerativeModel')
    def test_initialization_success(self, mock_model_class):
        """初期化成功のテスト"""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = MockConfig()
        client = GeminiClient(config)
        
        assert client.model == mock_model
        mock_model_class.assert_called_once_with("test-model")
    
    @patch('api_clients.genai.GenerativeModel')
    def test_generate_content_success(self, mock_model_class):
        """コンテンツ生成成功のテスト"""
        mock_response = Mock()
        mock_response.text = "Generated content"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        config = MockConfig()
        client = GeminiClient(config)
        
        result = client.generate_content("Test prompt")
        
        assert result == "Generated content"
        mock_model.generate_content.assert_called_once_with("Test prompt")
    
    @patch('api_clients.genai.GenerativeModel')
    def test_generate_content_no_response(self, mock_model_class):
        """レスポンスなしのコンテンツ生成テスト"""
        mock_model = Mock()
        mock_model.generate_content.return_value = None
        mock_model_class.return_value = mock_model
        
        config = MockConfig()
        client = GeminiClient(config)
        
        result = client.generate_content("Test prompt")
        
        assert result is None
    
    @patch('api_clients.genai.GenerativeModel')
    def test_generate_content_prompt_truncation(self, mock_model_class):
        """プロンプト切り詰めのテスト"""
        mock_response = Mock()
        mock_response.text = "Generated content"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        config = MockConfig()
        config.AI_TEXT_LIMIT = 10  # 短い制限
        client = GeminiClient(config)
        
        long_prompt = "A" * 20  # 制限を超える長さ
        result = client.generate_content(long_prompt)
        
        # 切り詰められたプロンプトで呼び出されることを確認
        called_prompt = mock_model.generate_content.call_args[0][0]
        assert len(called_prompt) <= 10

class TestAPIClientFactory:
    """APIClientFactory のテスト"""
    
    def test_create_yfinance_client(self):
        """YFinanceクライアント作成のテスト"""
        config = MockConfig()
        client = APIClientFactory.create_yfinance_client(config)
        
        assert isinstance(client, YFinanceClient)
        assert client.config == config
    
    def test_create_investpy_client(self):
        """Investpyクライアント作成のテスト"""
        config = MockConfig()
        client = APIClientFactory.create_investpy_client(config)
        
        assert isinstance(client, InvestpyClient)
        assert client.config == config
    
    @patch('api_clients.genai.GenerativeModel')
    def test_create_gemini_client(self, mock_model_class):
        """Geminiクライアント作成のテスト"""
        mock_model_class.return_value = Mock()
        
        config = MockConfig()
        client = APIClientFactory.create_gemini_client(config)
        
        assert isinstance(client, GeminiClient)
        assert client.config == config
    
    def test_create_clients_with_default_config(self):
        """デフォルト設定でのクライアント作成テスト"""
        # デフォルト設定でも作成できることを確認
        yf_client = APIClientFactory.create_yfinance_client()
        assert isinstance(yf_client, YFinanceClient)
        
        investpy_client = APIClientFactory.create_investpy_client()
        assert isinstance(investpy_client, InvestpyClient)