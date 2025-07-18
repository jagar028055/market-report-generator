"""
チャート生成の基底クラス
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import logging

from ..config import get_chart_config
from ..utils.exceptions import ChartGenerationError, FontError, ChartConfigurationError
from ..utils.error_handler import ErrorHandler


class BaseChartGenerator(ABC):
    """チャート生成の基底クラス"""
    
    def __init__(self, charts_dir: str = "charts", logger: Optional[logging.Logger] = None):
        self.charts_dir = Path(charts_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.config = get_chart_config()
        self.error_handler = ErrorHandler(self.logger)
        
        # チャートディレクトリを作成
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # 日本語フォントの設定
        self.japanese_font_path = self._setup_japanese_font()
        
        self.logger.info(f"Initialized {self.__class__.__name__} with output directory: {self.charts_dir}")
    
    def _setup_japanese_font(self) -> Optional[str]:
        """日本語フォントの設定"""
        try:
            # 利用可能なフォントパスを探す
            font_path = self.config.get_available_font_path()
            
            if font_path:
                # matplotlibに日本語フォントを設定
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.family'] = [font_name] + plt.rcParams['font.family']
                plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
                
                self.logger.info(f"Japanese font configured: {font_name}")
                return font_path
            else:
                self.logger.warning("No Japanese font found. Charts may not display Japanese text correctly.")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to setup Japanese font: {e}")
            return None
    
    @abstractmethod
    def generate_chart(self, data: Any, title: str, filename: str, **kwargs) -> Optional[str]:
        """チャートを生成する（サブクラスで実装）"""
        pass
    
    def validate_data(self, data: Any, data_type: str) -> bool:
        """データの妥当性をチェック"""
        if data is None:
            self.logger.error(f"{data_type} data is None")
            return False
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                self.logger.error(f"{data_type} DataFrame is empty")
                return False
            
            # 最小行数チェック
            if len(data) < 1:
                self.logger.error(f"{data_type} DataFrame has insufficient rows")
                return False
                
        elif isinstance(data, dict):
            if not data:
                self.logger.error(f"{data_type} dictionary is empty")
                return False
                
        return True
    
    def get_output_path(self, filename: str) -> Path:
        """出力パスを取得"""
        return self.charts_dir / filename
    
    def save_chart(self, figure, filename: str, format: str = None) -> Optional[str]:
        """チャートを保存"""
        try:
            if format is None:
                format = self.config.DEFAULT_OUTPUT_FORMAT
            
            # ファイル拡張子を調整
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = self.get_output_path(filename)
            
            if hasattr(figure, 'savefig'):
                # matplotlib figure
                figure.savefig(
                    output_path,
                    dpi=self.config.CHART_DPI,
                    format=format,
                    bbox_inches='tight'
                )
            else:
                # その他のfigureオブジェクト
                figure.write_html(str(output_path))
            
            self.logger.info(f"Chart saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'save_chart', 'filename': filename})
            return None
    
    def cleanup_chart(self, figure):
        """チャートリソースをクリーンアップ"""
        try:
            if hasattr(figure, 'close'):
                figure.close()
            elif hasattr(plt, 'close'):
                plt.close(figure)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup chart: {e}")
    
    def get_chart_style(self) -> Dict[str, Any]:
        """チャートスタイルを取得"""
        return {
            'figure_size': self.config.MATPLOTLIB_FIGURE_SIZE,
            'dpi': self.config.CHART_DPI,
            'font_family': plt.rcParams['font.family'],
            'colors': self.config.CANDLE_COLORS
        }
    
    def apply_chart_style(self, figure):
        """チャートにスタイルを適用"""
        try:
            if hasattr(figure, 'set_size_inches'):
                figure.set_size_inches(self.config.MATPLOTLIB_FIGURE_SIZE)
            
            if hasattr(figure, 'set_dpi'):
                figure.set_dpi(self.config.CHART_DPI)
                
        except Exception as e:
            self.logger.warning(f"Failed to apply chart style: {e}")
    
    def validate_chart_config(self):
        """チャート設定の妥当性をチェック"""
        try:
            # 基本設定のチェック
            if self.config.CHART_WIDTH <= 0 or self.config.CHART_HEIGHT <= 0:
                raise ChartConfigurationError("Chart dimensions must be positive")
            
            if self.config.CHART_DPI <= 0:
                raise ChartConfigurationError("Chart DPI must be positive")
            
            # 出力フォーマットのチェック
            if not self.config.is_output_format_supported(self.config.DEFAULT_OUTPUT_FORMAT):
                raise ChartConfigurationError(f"Unsupported output format: {self.config.DEFAULT_OUTPUT_FORMAT}")
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, {'operation': 'validate_chart_config'})
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """エラーの概要を取得"""
        return self.error_handler.get_error_summary()
    
    def clear_error_history(self):
        """エラー履歴をクリア"""
        self.error_handler.clear_history()
    
    def _handle_generation_error(self, error: Exception, context: str):
        """チャート生成エラーの処理"""
        self.error_handler.handle_error(error, {'context': context})
        
        if isinstance(error, FontError):
            self.logger.error(f"Font error in {context}: {error}")
        elif isinstance(error, ChartConfigurationError):
            self.logger.error(f"Configuration error in {context}: {error}")
        else:
            self.logger.error(f"Unexpected error in {context}: {error}")
    
    def _log_generation_result(self, result: Any, chart_type: str, success: bool = True):
        """チャート生成結果のログ"""
        if success:
            self.logger.info(f"Successfully generated {chart_type} chart")
        else:
            self.logger.error(f"Failed to generate {chart_type} chart")
    
    def get_supported_formats(self) -> List[str]:
        """サポートされているフォーマットのリストを取得"""
        return self.config.CHART_OUTPUT_FORMATS.copy()
    
    def get_default_format(self) -> str:
        """デフォルトフォーマットを取得"""
        return self.config.DEFAULT_OUTPUT_FORMAT
    
    def set_output_directory(self, directory: str):
        """出力ディレクトリを設定"""
        self.charts_dir = Path(directory)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory set to: {self.charts_dir}")
    
    def get_japanese_font_path(self) -> Optional[str]:
        """日本語フォントパスを取得"""
        return self.japanese_font_path
    
    def is_japanese_font_available(self) -> bool:
        """日本語フォントが利用可能かチェック"""
        return self.japanese_font_path is not None


class ChartGeneratorFactory:
    """チャート生成器のファクトリークラス"""
    
    _generators = {}
    
    @classmethod
    def register_generator(cls, name: str, generator_class: type):
        """ジェネレーターを登録"""
        cls._generators[name] = generator_class
    
    @classmethod
    def create_generator(cls, name: str, **kwargs) -> BaseChartGenerator:
        """ジェネレーターを作成"""
        if name not in cls._generators:
            raise ValueError(f"Unknown generator: {name}")
        
        generator_class = cls._generators[name]
        return generator_class(**kwargs)
    
    @classmethod
    def get_available_generators(cls) -> List[str]:
        """利用可能なジェネレーターのリストを取得"""
        return list(cls._generators.keys())


class ChartTemplate:
    """チャートテンプレートクラス"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    def apply_to_chart(self, chart_generator: BaseChartGenerator):
        """テンプレートをチャートジェネレーターに適用"""
        try:
            # テンプレート設定を適用
            for key, value in self.config.items():
                if hasattr(chart_generator, key):
                    setattr(chart_generator, key, value)
            
            return True
            
        except Exception as e:
            chart_generator.logger.error(f"Failed to apply template {self.name}: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """テンプレート設定を取得"""
        return self.config.copy()


class ChartStyleManager:
    """チャートスタイル管理クラス"""
    
    def __init__(self):
        self.styles = {}
    
    def register_style(self, name: str, style_config: Dict[str, Any]):
        """スタイルを登録"""
        self.styles[name] = style_config
    
    def get_style(self, name: str) -> Dict[str, Any]:
        """スタイルを取得"""
        return self.styles.get(name, {})
    
    def apply_style(self, chart_generator: BaseChartGenerator, style_name: str):
        """スタイルを適用"""
        style = self.get_style(style_name)
        if style:
            for key, value in style.items():
                if hasattr(chart_generator, key):
                    setattr(chart_generator, key, value)
    
    def get_available_styles(self) -> List[str]:
        """利用可能なスタイルのリストを取得"""
        return list(self.styles.keys())