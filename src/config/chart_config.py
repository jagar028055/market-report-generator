"""
チャート生成関連の設定
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from pathlib import Path

from .base_config import BaseConfig, ConfigValidator


@dataclass
class ChartConfig(BaseConfig):
    """チャート生成設定"""
    
    # 基本チャート設定
    CHART_WIDTH: int = 1200
    CHART_HEIGHT: int = 600
    CHART_DPI: int = 150
    
    # 図形サイズ設定
    MATPLOTLIB_FIGURE_SIZE: Tuple[int, int] = (12, 6)
    
    # Plotly設定
    PLOTLY_JS_SOURCE: str = 'cdn'
    PLOTLY_TEMPLATE: str = 'simple_white'
    
    # 日本語フォント設定
    JAPANESE_FONT_PATHS: List[str] = field(default_factory=lambda: [
        '/System/Library/Fonts/ヒラギノ角ゴ ProN W3.ttc',
        '/System/Library/Fonts/Hiragino Sans/Hiragino Sans W3.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴ ProN W3.ttc',
        '/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/Library/Fonts/Osaka.ttf',
        '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
        '/System/Library/Fonts/ヒラギノ明朝 ProN W3.ttc'
    ])
    
    # 移動平均設定
    MOVING_AVERAGES: Dict[str, Dict] = field(default_factory=lambda: {
        "short": {"period": 25, "color": "blue", "label": "MA25"},
        "medium": {"period": 50, "color": "orange", "label": "MA50"},
        "long": {"period": 75, "color": "red", "label": "MA75"}
    })
    
    # デフォルトで表示する移動平均
    DEFAULT_MA_DISPLAY: List[str] = field(default_factory=lambda: ["short", "long"])
    
    # 移動平均タイプ設定
    MA_TYPES: Dict[str, str] = field(default_factory=lambda: {
        "SMA": "Simple Moving Average",
        "EMA": "Exponential Moving Average",
        "WMA": "Weighted Moving Average"
    })
    
    # デフォルト移動平均タイプ
    DEFAULT_MA_TYPE: str = "SMA"
    
    # ローソク足の色設定
    CANDLE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "up_fill": "white",
        "down_fill": "black",
        "up_line": "black",
        "down_line": "black",
        "wick": "black"
    })
    
    # チャート出力設定
    CHART_OUTPUT_FORMATS: List[str] = field(default_factory=lambda: ["html", "png", "svg"])
    DEFAULT_OUTPUT_FORMAT: str = "html"
    
    # セクターパフォーマンスチャート設定
    SECTOR_CHART_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "positive": "green",
        "negative": "red",
        "neutral": "gray"
    })
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'charts' in config_dict:
            chart_config = config_dict['charts']
            
            # 基本設定
            if 'width' in chart_config:
                self.CHART_WIDTH = chart_config['width']
            if 'height' in chart_config:
                self.CHART_HEIGHT = chart_config['height']
            if 'dpi' in chart_config:
                self.CHART_DPI = chart_config['dpi']
            
            # Plotly設定
            if 'plotly_js_source' in chart_config:
                self.PLOTLY_JS_SOURCE = chart_config['plotly_js_source']
            if 'plotly_template' in chart_config:
                self.PLOTLY_TEMPLATE = chart_config['plotly_template']
            
            # フォント設定
            if 'japanese_font_paths' in chart_config:
                self.JAPANESE_FONT_PATHS = chart_config['japanese_font_paths']
            
            # 移動平均設定
            if 'moving_averages' in chart_config:
                self.MOVING_AVERAGES = chart_config['moving_averages']
            if 'default_ma_display' in chart_config:
                self.DEFAULT_MA_DISPLAY = chart_config['default_ma_display']
            if 'default_ma_type' in chart_config:
                self.DEFAULT_MA_TYPE = chart_config['default_ma_type']
            
            # 色設定
            if 'candle_colors' in chart_config:
                self.CANDLE_COLORS.update(chart_config['candle_colors'])
            if 'sector_chart_colors' in chart_config:
                self.SECTOR_CHART_COLORS.update(chart_config['sector_chart_colors'])
            
            # 出力設定
            if 'output_formats' in chart_config:
                self.CHART_OUTPUT_FORMATS = chart_config['output_formats']
            if 'default_output_format' in chart_config:
                self.DEFAULT_OUTPUT_FORMAT = chart_config['default_output_format']
            
            # matplotlib設定
            if 'matplotlib_figure_size' in chart_config:
                self.MATPLOTLIB_FIGURE_SIZE = tuple(chart_config['matplotlib_figure_size'])
    
    def _validate_configuration(self):
        """設定値の検証"""
        # 基本設定の検証
        ConfigValidator.validate_positive_integer(self.CHART_WIDTH, "CHART_WIDTH")
        ConfigValidator.validate_positive_integer(self.CHART_HEIGHT, "CHART_HEIGHT")
        ConfigValidator.validate_positive_integer(self.CHART_DPI, "CHART_DPI")
        
        # 文字列設定の検証
        ConfigValidator.validate_string(self.PLOTLY_JS_SOURCE, "PLOTLY_JS_SOURCE")
        ConfigValidator.validate_string(self.PLOTLY_TEMPLATE, "PLOTLY_TEMPLATE")
        ConfigValidator.validate_string(self.DEFAULT_MA_TYPE, "DEFAULT_MA_TYPE")
        ConfigValidator.validate_string(self.DEFAULT_OUTPUT_FORMAT, "DEFAULT_OUTPUT_FORMAT")
        
        # 辞書設定の検証
        ConfigValidator.validate_dict(self.MOVING_AVERAGES, "MOVING_AVERAGES", min_items=1)
        ConfigValidator.validate_dict(self.MA_TYPES, "MA_TYPES", min_items=1)
        ConfigValidator.validate_dict(self.CANDLE_COLORS, "CANDLE_COLORS", min_items=1)
        ConfigValidator.validate_dict(self.SECTOR_CHART_COLORS, "SECTOR_CHART_COLORS", min_items=1)
        
        # リスト設定の検証
        ConfigValidator.validate_list(self.JAPANESE_FONT_PATHS, "JAPANESE_FONT_PATHS", min_items=1)
        ConfigValidator.validate_list(self.DEFAULT_MA_DISPLAY, "DEFAULT_MA_DISPLAY", min_items=1)
        ConfigValidator.validate_list(self.CHART_OUTPUT_FORMATS, "CHART_OUTPUT_FORMATS", min_items=1)
        
        # 選択肢の検証
        ConfigValidator.validate_choice(self.DEFAULT_MA_TYPE, list(self.MA_TYPES.keys()), "DEFAULT_MA_TYPE")
        ConfigValidator.validate_choice(self.DEFAULT_OUTPUT_FORMAT, self.CHART_OUTPUT_FORMATS, "DEFAULT_OUTPUT_FORMAT")
        
        # 移動平均設定の詳細検証
        for ma_key in self.DEFAULT_MA_DISPLAY:
            if ma_key not in self.MOVING_AVERAGES:
                raise ValueError(f"DEFAULT_MA_DISPLAY contains unknown key: {ma_key}")
        
        for ma_key, ma_config in self.MOVING_AVERAGES.items():
            if not isinstance(ma_config, dict):
                raise ValueError(f"MOVING_AVERAGES['{ma_key}'] must be a dictionary")
            
            required_keys = ['period', 'color', 'label']
            for key in required_keys:
                if key not in ma_config:
                    raise ValueError(f"MOVING_AVERAGES['{ma_key}'] missing required key: {key}")
            
            if not isinstance(ma_config['period'], int) or ma_config['period'] <= 0:
                raise ValueError(f"MOVING_AVERAGES['{ma_key}']['period'] must be a positive integer")
        
        # matplotlib図形サイズの検証
        if not isinstance(self.MATPLOTLIB_FIGURE_SIZE, tuple) or len(self.MATPLOTLIB_FIGURE_SIZE) != 2:
            raise ValueError("MATPLOTLIB_FIGURE_SIZE must be a tuple of two numbers")
        
        for size in self.MATPLOTLIB_FIGURE_SIZE:
            if not isinstance(size, (int, float)) or size <= 0:
                raise ValueError("MATPLOTLIB_FIGURE_SIZE values must be positive numbers")
    
    def get_moving_average_config(self, ma_key: str) -> Dict[str, Any]:
        """移動平均の設定を取得"""
        if ma_key not in self.MOVING_AVERAGES:
            raise ValueError(f"Moving average key '{ma_key}' not found")
        return self.MOVING_AVERAGES[ma_key]
    
    def add_moving_average(self, key: str, period: int, color: str, label: str):
        """移動平均設定を追加"""
        self.MOVING_AVERAGES[key] = {
            "period": period,
            "color": color,
            "label": label
        }
    
    def remove_moving_average(self, key: str):
        """移動平均設定を削除"""
        if key in self.MOVING_AVERAGES:
            del self.MOVING_AVERAGES[key]
        
        # デフォルト表示リストからも削除
        if key in self.DEFAULT_MA_DISPLAY:
            self.DEFAULT_MA_DISPLAY.remove(key)
    
    def get_available_font_path(self) -> str:
        """利用可能な日本語フォントパスを取得"""
        for font_path in self.JAPANESE_FONT_PATHS:
            if Path(font_path).exists():
                return font_path
        return None
    
    def is_output_format_supported(self, format_name: str) -> bool:
        """出力フォーマットがサポートされているかチェック"""
        return format_name in self.CHART_OUTPUT_FORMATS


@dataclass
class AIConfig(BaseConfig):
    """AI関連設定"""
    
    # Gemini設定
    GEMINI_PREFERRED_MODELS: List[str] = field(default_factory=lambda: [
        'models/gemini-2.5-flash-lite-preview-06-17',
        'models/gemini-2.5-flash-preview-05-20',
        'models/gemini-1.5-flash-latest'
    ])
    
    # テキスト処理設定
    AI_TEXT_LIMIT: int = 1800
    MAX_TOKENS: int = 8192
    TEMPERATURE: float = 0.7
    
    # リトライ設定
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_DELAY: int = 5
    
    # レスポンス設定
    RESPONSE_TIMEOUT: int = 30
    MAX_RESPONSE_LENGTH: int = 5000
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新"""
        if 'ai' in config_dict:
            ai_config = config_dict['ai']
            
            # モデル設定
            if 'preferred_models' in ai_config:
                self.GEMINI_PREFERRED_MODELS = ai_config['preferred_models']
            
            # テキスト処理設定
            if 'text_limit' in ai_config:
                self.AI_TEXT_LIMIT = ai_config['text_limit']
            if 'max_tokens' in ai_config:
                self.MAX_TOKENS = ai_config['max_tokens']
            if 'temperature' in ai_config:
                self.TEMPERATURE = ai_config['temperature']
            
            # リトライ設定
            if 'api_retry_attempts' in ai_config:
                self.API_RETRY_ATTEMPTS = ai_config['api_retry_attempts']
            if 'api_retry_delay' in ai_config:
                self.API_RETRY_DELAY = ai_config['api_retry_delay']
            
            # レスポンス設定
            if 'response_timeout' in ai_config:
                self.RESPONSE_TIMEOUT = ai_config['response_timeout']
            if 'max_response_length' in ai_config:
                self.MAX_RESPONSE_LENGTH = ai_config['max_response_length']
    
    def _validate_configuration(self):
        """設定値の検証"""
        # リスト設定の検証
        ConfigValidator.validate_list(self.GEMINI_PREFERRED_MODELS, "GEMINI_PREFERRED_MODELS", min_items=1)
        
        # 整数設定の検証
        ConfigValidator.validate_positive_integer(self.AI_TEXT_LIMIT, "AI_TEXT_LIMIT")
        ConfigValidator.validate_positive_integer(self.MAX_TOKENS, "MAX_TOKENS")
        ConfigValidator.validate_positive_integer(self.API_RETRY_ATTEMPTS, "API_RETRY_ATTEMPTS")
        ConfigValidator.validate_positive_integer(self.API_RETRY_DELAY, "API_RETRY_DELAY")
        ConfigValidator.validate_positive_integer(self.RESPONSE_TIMEOUT, "RESPONSE_TIMEOUT")
        ConfigValidator.validate_positive_integer(self.MAX_RESPONSE_LENGTH, "MAX_RESPONSE_LENGTH")
        
        # 浮動小数点設定の検証
        if not isinstance(self.TEMPERATURE, (int, float)) or not (0 <= self.TEMPERATURE <= 2):
            raise ValueError("TEMPERATURE must be a number between 0 and 2")
        
        # モデル名の検証
        for model in self.GEMINI_PREFERRED_MODELS:
            if not isinstance(model, str) or not model.startswith('models/'):
                raise ValueError(f"Invalid model name format: {model}")
    
    def get_primary_model(self) -> str:
        """プライマリモデルを取得"""
        return self.GEMINI_PREFERRED_MODELS[0]
    
    def add_model(self, model_name: str):
        """モデルを追加"""
        if model_name not in self.GEMINI_PREFERRED_MODELS:
            self.GEMINI_PREFERRED_MODELS.append(model_name)
    
    def remove_model(self, model_name: str):
        """モデルを削除"""
        if model_name in self.GEMINI_PREFERRED_MODELS:
            self.GEMINI_PREFERRED_MODELS.remove(model_name)