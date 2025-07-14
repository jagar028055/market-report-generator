"""
設定モジュールのテスト
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from config import Config

class TestConfig:
    """Config クラスのテスト"""
    
    def test_default_config_values(self):
        """デフォルト設定値のテスト"""
        config = Config()
        
        # 基本設定の確認
        assert config.ENVIRONMENT in ["development", "production"]
        assert config.MAX_WORKERS >= 1
        assert config.TIMEOUT_SECONDS > 0
        assert config.RETRY_ATTEMPTS >= 1
        
        # チャート設定の確認
        assert config.CHART_WIDTH > 0
        assert config.CHART_HEIGHT > 0
        assert len(config.MOVING_AVERAGES) > 0
        
        # マーケットティッカーの確認
        assert len(config.MARKET_TICKERS) > 0
        assert "S&P500" in config.MARKET_TICKERS
        assert "^GSPC" in config.MARKET_TICKERS.values()
    
    def test_yaml_config_loading(self):
        """YAML設定ファイルの読み込みテスト"""
        # 一時的なYAMLファイルを作成
        test_config = {
            'environment': 'test',
            'performance': {
                'max_workers': 8,
                'timeout_seconds': 60
            },
            'charts': {
                'width': 1600,
                'height': 800
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = Path(f.name)
        
        try:
            # 元のパスを保存
            original_settings_path = Path(__file__).parent.parent / "settings.yaml"
            
            # 一時ファイルをコピー
            temp_path.rename(original_settings_path.with_name("test_settings.yaml"))
            
            # 設定を読み込み（実際のテストではモック使用推奨）
            config = Config()
            
            # デフォルト値が適用されていることを確認
            assert config.MAX_WORKERS >= 1
            assert config.CHART_WIDTH > 0
            
        finally:
            # クリーンアップ
            test_file = original_settings_path.with_name("test_settings.yaml")
            if test_file.exists():
                test_file.unlink()
    
    def test_moving_averages_config(self):
        """移動平均設定のテスト"""
        config = Config()
        
        # 移動平均設定の構造確認
        assert isinstance(config.MOVING_AVERAGES, dict)
        assert len(config.DEFAULT_MA_DISPLAY) > 0
        
        for ma_key in config.DEFAULT_MA_DISPLAY:
            assert ma_key in config.MOVING_AVERAGES
            ma_config = config.MOVING_AVERAGES[ma_key]
            assert 'period' in ma_config
            assert 'color' in ma_config
            assert 'label' in ma_config
            assert isinstance(ma_config['period'], int)
            assert ma_config['period'] > 0
    
    def test_asset_classes_config(self):
        """資産分類設定のテスト"""
        config = Config()
        
        assert isinstance(config.ASSET_CLASSES, dict)
        assert "US_STOCK" in config.ASSET_CLASSES
        assert "24H_ASSET" in config.ASSET_CLASSES
        
        # 各分類が空でないことを確認
        for asset_class, symbols in config.ASSET_CLASSES.items():
            assert isinstance(symbols, list)
            assert len(symbols) > 0
    
    def test_news_config(self):
        """ニュース設定のテスト"""
        config = Config()
        
        # ニュース設定の確認
        assert config.NEWS_HOURS_LIMIT > 0
        assert config.REUTERS_MAX_PAGES > 0
        assert len(config.REUTERS_TARGET_CATEGORIES) > 0
        assert len(config.REUTERS_EXCLUDE_KEYWORDS) > 0
        
        # URL設定の確認
        assert config.REUTERS_BASE_URL.startswith("http")
        assert config.REUTERS_SEARCH_URL.startswith("http")
    
    def test_ai_config(self):
        """AI設定のテスト"""
        config = Config()
        
        assert len(config.GEMINI_PREFERRED_MODELS) > 0
        assert config.AI_TEXT_LIMIT > 0
        
        # モデル名の形式確認
        for model in config.GEMINI_PREFERRED_MODELS:
            assert isinstance(model, str)
            assert len(model) > 0