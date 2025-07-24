"""
チャート生成モジュール

従来の巨大なChartGeneratorクラスを機能別に分割したものです。
各専用クラスは特定の責任を持ち、テストやメンテナンスが容易になっています。
"""

from .base_chart_generator import BaseChartGenerator, ChartGeneratorFactory, ChartTemplate, ChartStyleManager
from .candlestick_chart_generator import CandlestickChartGenerator
from .sector_chart_generator import SectorChartGenerator

# 下位互換性のための統合クラス
class ChartGenerator:
    """
    下位互換性のための統合チャートジェネレーター
    
    従来のChartGeneratorクラスと同じインターフェースを提供しつつ、
    内部的には分割されたクラスを使用します。
    """
    
    def __init__(self, charts_dir: str = "charts", config=None):
        self.charts_dir = charts_dir
        self.config = config
        
        # 各専用ジェネレーターを初期化
        self.candlestick_generator = CandlestickChartGenerator(charts_dir)
        self.sector_generator = SectorChartGenerator(charts_dir)
        
        # 下位互換性のための属性
        self.japanese_font_path = self.candlestick_generator.japanese_font_path
        
        # スタイルマネージャーを初期化
        self.style_manager = ChartStyleManager()
        self._register_default_styles()
    
    def _register_default_styles(self):
        """デフォルトスタイルを登録"""
        self.style_manager.register_style("default", {
            "template": "simple_white",
            "color_scheme": "default"
        })
        
        self.style_manager.register_style("dark", {
            "template": "plotly_dark",
            "color_scheme": "dark"
        })
    
    def generate_intraday_chart_interactive(self, data, ticker_name: str, filename: str):
        """イントラデイチャートをインタラクティブHTMLとして生成"""
        return self.candlestick_generator.generate_intraday_chart(data, ticker_name, filename)
    
    def generate_longterm_chart_interactive(self, data, ticker_name: str, filename: str, 
                                          ma_keys=None, ma_type=None):
        """長期チャートをインタラクティブHTMLとして生成"""
        return self.candlestick_generator.generate_longterm_chart(
            data, ticker_name, filename, ma_keys, ma_type
        )
    
    def generate_intraday_chart(self, data, ticker_name: str, filename: str):
        """イントラデイチャートを静的画像として生成"""
        return self.candlestick_generator.generate_static_chart(
            data, f"{ticker_name} Intraday Chart (Tokyo Time)", filename,
            chart_type='static'
        )
    
    def generate_longterm_chart(self, data, ticker_name: str, filename: str, 
                               ma_keys=None, ma_type=None):
        """長期チャートを静的画像として生成"""
        # 移動平均の情報をタイトルに追加
        ma_info = ""
        if ma_keys:
            ma_config = self.candlestick_generator.config
            ma_labels = [
                ma_config.MOVING_AVERAGES[key]["label"] 
                for key in ma_keys 
                if key in ma_config.MOVING_AVERAGES
            ]
            if ma_labels:
                ma_type_display = ma_type or ma_config.DEFAULT_MA_TYPE
                ma_info = f" ({ma_type_display}: {', '.join(ma_labels)})"
        
        title = f"{ticker_name} Long-Term Chart (1 Year){ma_info}"
        
        return self.candlestick_generator.generate_static_chart(
            data, title, filename, 
            chart_type='static', ma_keys=ma_keys, ma_type=ma_type
        )
    
    def generate_intraday_chart_static(self, data, ticker_name: str, filename: str):
        """イントラデイチャートの静的版（PNG）を生成"""
        return self.candlestick_generator.generate_intraday_chart(
            data, ticker_name, filename, chart_type='static'
        )
    
    def generate_longterm_chart_static(self, data, ticker_name: str, filename: str, 
                                     ma_keys=None, ma_type=None):
        """長期チャートの静的版（PNG）を生成"""
        return self.candlestick_generator.generate_longterm_chart(
            data, ticker_name, filename, chart_type='static', 
            ma_keys=ma_keys, ma_type=ma_type
        )
    
    def generate_sector_performance_chart(self, data: dict, filename: str):
        """セクター別ETFの変化率チャートを生成"""
        return self.sector_generator.generate_sector_performance_chart(data, filename)
    
    def _setup_japanese_font(self):
        """日本語フォントの設定（下位互換性）"""
        return self.candlestick_generator._setup_japanese_font()
    
    def _get_japanese_font_path(self):
        """日本語フォントパスを取得（下位互換性）"""
        return self.candlestick_generator.get_japanese_font_path()
    
    def get_error_summary(self):
        """すべてのジェネレーターのエラー概要を取得"""
        return {
            "candlestick": self.candlestick_generator.get_error_summary(),
            "sector": self.sector_generator.get_error_summary()
        }
    
    def clear_error_history(self):
        """すべてのジェネレーターのエラー履歴をクリア"""
        self.candlestick_generator.clear_error_history()
        self.sector_generator.clear_error_history()
    
    def set_output_directory(self, directory: str):
        """出力ディレクトリを設定"""
        self.charts_dir = directory
        self.candlestick_generator.set_output_directory(directory)
        self.sector_generator.set_output_directory(directory)
    
    def apply_style(self, style_name: str):
        """スタイルを適用"""
        self.style_manager.apply_style(self.candlestick_generator, style_name)
        self.style_manager.apply_style(self.sector_generator, style_name)
    
    def get_available_styles(self):
        """利用可能なスタイルのリストを取得"""
        return self.style_manager.get_available_styles()
    
    def create_custom_template(self, name: str, config: dict):
        """カスタムテンプレートを作成"""
        template = ChartTemplate(name, config)
        return template
    
    def validate_all_configs(self):
        """すべてのジェネレーターの設定を検証"""
        results = []
        
        try:
            results.append(("candlestick", self.candlestick_generator.validate_chart_config()))
        except Exception as e:
            results.append(("candlestick", False))
        
        try:
            results.append(("sector", self.sector_generator.validate_chart_config()))
        except Exception as e:
            results.append(("sector", False))
        
        return results


# ファクトリーパターンを使用した取得
def create_chart_generator(generator_type: str = "integrated", **kwargs):
    """
    チャートジェネレーターを作成
    
    Args:
        generator_type: ジェネレーターの種類 ("integrated", "candlestick", "sector")
        **kwargs: ジェネレーター固有の引数
    
    Returns:
        指定されたタイプのチャートジェネレーター
    """
    if generator_type == "integrated":
        return ChartGenerator(**kwargs)
    elif generator_type in ["candlestick", "sector"]:
        return ChartGeneratorFactory.create_generator(generator_type, **kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


# 使用例とユーティリティ関数
def generate_all_charts(chart_data: dict, charts_dir: str = "charts"):
    """すべてのチャートを一括生成"""
    
    generator = ChartGenerator(charts_dir)
    generated_charts = {"Intraday": [], "Long-Term": []}
    
    try:
        for name, data_set in chart_data.items():
            # イントラデイチャート
            if "intraday" in data_set and not data_set["intraday"].empty:
                intraday_filename = f"{name.replace(' ', '_')}_intraday.html"
                intraday_path = generator.generate_intraday_chart_interactive(
                    data_set["intraday"], name, intraday_filename
                )
                
                if intraday_path:
                    sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    generated_charts["Intraday"].append({
                        "id": f"{sanitized_name}-intraday",
                        "name": name,
                        "path": f"charts/{intraday_filename}",
                        "interactive": True
                    })
            
            # 長期チャート
            if "longterm" in data_set and not data_set["longterm"].empty:
                longterm_filename = f"{name.replace(' ', '_')}_longterm.html"
                longterm_path = generator.generate_longterm_chart_interactive(
                    data_set["longterm"], name, longterm_filename
                )
                
                if longterm_path:
                    sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    generated_charts["Long-Term"].append({
                        "id": f"{sanitized_name}-longterm",
                        "name": name,
                        "path": f"charts/{longterm_filename}",
                        "interactive": True
                    })
        
        return generated_charts
        
    except Exception as e:
        raise Exception(f"Failed to generate all charts: {e}")


def generate_sector_chart(sector_data: dict, charts_dir: str = "charts", filename: str = "sector_performance_chart.html"):
    """セクターチャートを生成"""
    
    generator = SectorChartGenerator(charts_dir)
    
    try:
        # データをソート
        sorted_data = dict(sorted(
            sector_data.items(),
            key=lambda item: item[1] if item[1] is not None else -float('inf'),
            reverse=True
        ))
        
        chart_path = generator.generate_sector_performance_chart(sorted_data, filename)
        
        return chart_path
        
    except Exception as e:
        raise Exception(f"Failed to generate sector chart: {e}")


def get_chart_capabilities():
    """チャート生成機能の一覧を取得"""
    return {
        "generators": ChartGeneratorFactory.get_available_generators(),
        "formats": ["html", "png", "svg", "pdf"],
        "chart_types": ["candlestick", "sector", "comparison"],
        "features": [
            "Interactive charts",
            "Moving averages",
            "Japanese font support",
            "Custom styling",
            "Error handling",
            "Template system"
        ]
    }


# モジュールの公開API
__all__ = [
    'BaseChartGenerator',
    'ChartGeneratorFactory',
    'ChartTemplate',
    'ChartStyleManager',
    'CandlestickChartGenerator',
    'SectorChartGenerator',
    'ChartGenerator',
    'create_chart_generator',
    'generate_all_charts',
    'generate_sector_chart',
    'get_chart_capabilities'
]


# 古いインポートパターンとの互換性を保つ
def ChartGenerator_legacy(charts_dir: str = "charts", config=None):
    """
    従来のインポートパターンとの互換性を保つ関数
    
    使用例:
        from src.chart_generators import ChartGenerator_legacy as ChartGenerator
        generator = ChartGenerator()
    """
    import warnings
    warnings.warn(
        "Direct ChartGenerator import is deprecated. Use create_chart_generator() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ChartGenerator(charts_dir, config)