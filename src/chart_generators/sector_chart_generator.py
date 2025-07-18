"""
セクターパフォーマンスチャート専用生成クラス
"""

import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .base_chart_generator import BaseChartGenerator
from ..utils.exceptions import ChartGenerationError
from ..utils.error_handler import with_error_handling


class SectorChartGenerator(BaseChartGenerator):
    """セクターパフォーマンスチャート専用生成クラス"""
    
    def __init__(self, charts_dir: str = "charts", logger: Optional[Any] = None):
        super().__init__(charts_dir, logger)
        self.logger.info("Initialized SectorChartGenerator")
    
    @with_error_handling()
    def generate_chart(self, data: Dict[str, Any], title: str, filename: str, **kwargs) -> Optional[str]:
        """セクターパフォーマンスチャートを生成"""
        chart_type = kwargs.get('chart_type', 'interactive')
        
        if chart_type == 'interactive':
            return self.generate_interactive_chart(data, title, filename, **kwargs)
        else:
            return self.generate_static_chart(data, title, filename, **kwargs)
    
    def generate_interactive_chart(
        self, 
        data: Dict[str, Any], 
        title: str, 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """インタラクティブなセクターパフォーマンスチャートを生成"""
        
        if not self.validate_data(data, "sector_performance"):
            return None
        
        try:
            # データを準備
            chart_data = self._prepare_chart_data(data)
            
            if not chart_data:
                self.logger.warning("No valid sector data to chart")
                return None
            
            # 横棒グラフを作成
            fig = self._create_plotly_bar_chart(chart_data, title, **kwargs)
            
            # ファイルに保存
            output_path = self.get_output_path(filename)
            pio.write_html(fig, file=str(output_path), include_plotlyjs=self.config.PLOTLY_JS_SOURCE, full_html=True)
            
            self._log_generation_result(output_path, "interactive sector", True)
            return str(output_path)
            
        except Exception as e:
            self._handle_generation_error(e, "interactive sector chart generation")
            return None
    
    def generate_static_chart(
        self, 
        data: Dict[str, Any], 
        title: str, 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """静的なセクターパフォーマンスチャートを生成"""
        
        if not self.validate_data(data, "sector_performance"):
            return None
        
        try:
            # データを準備
            chart_data = self._prepare_chart_data(data)
            
            if not chart_data:
                self.logger.warning("No valid sector data to chart")
                return None
            
            # matplotlib横棒グラフを作成
            fig = self._create_matplotlib_bar_chart(chart_data, title, **kwargs)
            
            # ファイルに保存
            output_path = self.save_chart(fig, filename, 'png')
            
            # リソースをクリーンアップ
            self.cleanup_chart(fig)
            
            self._log_generation_result(output_path, "static sector", True)
            return output_path
            
        except Exception as e:
            self._handle_generation_error(e, "static sector chart generation")
            return None
    
    def generate_sector_performance_chart(
        self, 
        data: Dict[str, Any], 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """セクター別ETFの変化率チャートを生成"""
        
        title = kwargs.get('title', '米国セクターETF変化率')
        
        return self.generate_interactive_chart(data, title, filename, **kwargs)
    
    def _prepare_chart_data(self, data: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """チャート用データを準備"""
        
        chart_data = []
        
        for name, value in data.items():
            try:
                # 数値データのみを処理
                if isinstance(value, (int, float)) and not np.isnan(value):
                    # セクター名を短縮
                    short_name = self._shorten_sector_name(name)
                    
                    # 色を決定
                    color = self._determine_color(value)
                    
                    chart_data.append((short_name, value, color))
                    
            except (ValueError, TypeError):
                self.logger.warning(f"Skipping invalid sector data: {name} = {value}")
                continue
        
        # 値でソート（降順）
        chart_data.sort(key=lambda x: x[1], reverse=True)
        
        return chart_data
    
    def _shorten_sector_name(self, name: str) -> str:
        """セクター名を短縮"""
        
        # 一般的な短縮パターン
        replacements = {
            " Select Sector SPDR Fund": "",
            " Select Sector PDR Fund": "",
            " Select Sector": "",
            " SPDR Fund": "",
            " PDR Fund": "",
            "Technology": "Tech",
            "Information Technology": "Tech",
            "Communication Services": "Comm",
            "Consumer Discretionary": "Cons Disc",
            "Consumer Staples": "Cons Stap",
            "Financial": "Finance",
            "Health Care": "Healthcare",
            "Real Estate": "Real Est",
            "Materials": "Material",
            "Utilities": "Utility"
        }
        
        short_name = name
        for old, new in replacements.items():
            short_name = short_name.replace(old, new)
        
        return short_name.strip()
    
    def _determine_color(self, value: float) -> str:
        """値に基づいて色を決定"""
        
        if value > 0:
            return self.config.SECTOR_CHART_COLORS.get('positive', 'green')
        elif value < 0:
            return self.config.SECTOR_CHART_COLORS.get('negative', 'red')
        else:
            return self.config.SECTOR_CHART_COLORS.get('neutral', 'gray')
    
    def _create_plotly_bar_chart(
        self, 
        chart_data: List[Tuple[str, float, str]], 
        title: str, 
        **kwargs
    ) -> go.Figure:
        """Plotly横棒グラフを作成"""
        
        sectors = [item[0] for item in chart_data]
        values = [item[1] for item in chart_data]
        colors = [item[2] for item in chart_data]
        
        # テキストラベルを作成
        text_labels = [f"{value:.2f}%" for value in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=sectors,
            orientation='h',
            marker_color=colors,
            text=text_labels,
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>変化率: %{x:.2f}%<extra></extra>'
        ))
        
        # レイアウトを設定
        fig.update_layout(
            title=title,
            xaxis_title='変化率 (%)',
            yaxis_title='セクター',
            template=self.config.PLOTLY_TEMPLATE,
            height=max(400, len(sectors) * 40 + 100),
            width=self.config.CHART_WIDTH,
            yaxis=dict(autorange='reversed'),  # 上位から表示
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # グリッドラインを追加
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def _create_matplotlib_bar_chart(
        self, 
        chart_data: List[Tuple[str, float, str]], 
        title: str, 
        **kwargs
    ) -> plt.Figure:
        """matplotlib横棒グラフを作成"""
        
        sectors = [item[0] for item in chart_data]
        values = [item[1] for item in chart_data]
        colors = [item[2] for item in chart_data]
        
        # 図のサイズを調整
        fig_height = max(6, len(sectors) * 0.4 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # 横棒グラフを作成
        bars = ax.barh(sectors, values, color=colors, alpha=0.7, edgecolor='black')
        
        # バーに値を表示
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_width() + (0.1 if value >= 0 else -0.1),
                bar.get_y() + bar.get_height()/2,
                f'{value:.2f}%',
                ha='left' if value >= 0 else 'right',
                va='center',
                fontsize=9
            )
        
        # グラフの装飾
        ax.set_xlabel('変化率 (%)')
        ax.set_ylabel('セクター')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Y軸を反転（上位から表示）
        ax.invert_yaxis()
        
        # 0線を強調
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        return fig
    
    def validate_sector_data(self, data: Dict[str, Any]) -> bool:
        """セクターデータの特別な検証"""
        
        if not super().validate_data(data, "sector_performance"):
            return False
        
        # 数値データの存在確認
        numeric_data = []
        for name, value in data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_data.append((name, value))
        
        if not numeric_data:
            self.logger.error("No valid numeric sector data found")
            return False
        
        self.logger.info(f"Found {len(numeric_data)} valid sector data points")
        return True
    
    def create_sector_comparison_chart(
        self, 
        data_sets: Dict[str, Dict[str, Any]], 
        title: str, 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """複数のセクターデータセットを比較するチャートを作成"""
        
        try:
            # データを準備
            comparison_data = self._prepare_comparison_data(data_sets)
            
            if not comparison_data:
                self.logger.warning("No valid comparison data")
                return None
            
            # 比較チャートを作成
            fig = self._create_comparison_chart(comparison_data, title, **kwargs)
            
            # ファイルに保存
            output_path = self.get_output_path(filename)
            pio.write_html(fig, file=str(output_path), include_plotlyjs=self.config.PLOTLY_JS_SOURCE, full_html=True)
            
            self._log_generation_result(output_path, "sector comparison", True)
            return str(output_path)
            
        except Exception as e:
            self._handle_generation_error(e, "sector comparison chart generation")
            return None
    
    def _prepare_comparison_data(self, data_sets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """比較用データを準備"""
        
        comparison_data = {}
        
        # すべてのセクター名を取得
        all_sectors = set()
        for dataset in data_sets.values():
            all_sectors.update(dataset.keys())
        
        # 各セクターのデータを整理
        for sector in all_sectors:
            sector_data = {}
            for period, dataset in data_sets.items():
                value = dataset.get(sector, None)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    sector_data[period] = value
            
            if sector_data:
                short_name = self._shorten_sector_name(sector)
                comparison_data[short_name] = sector_data
        
        return comparison_data
    
    def _create_comparison_chart(
        self, 
        comparison_data: Dict[str, Any], 
        title: str, 
        **kwargs
    ) -> go.Figure:
        """比較チャートを作成"""
        
        fig = go.Figure()
        
        # 各期間のデータを追加
        periods = list(next(iter(comparison_data.values())).keys())
        
        for period in periods:
            sectors = []
            values = []
            
            for sector, data in comparison_data.items():
                if period in data:
                    sectors.append(sector)
                    values.append(data[period])
            
            fig.add_trace(go.Bar(
                name=period,
                x=sectors,
                y=values,
                text=[f"{v:.2f}%" for v in values],
                textposition='auto'
            ))
        
        # レイアウトを設定
        fig.update_layout(
            title=title,
            xaxis_title='セクター',
            yaxis_title='変化率 (%)',
            template=self.config.PLOTLY_TEMPLATE,
            height=self.config.CHART_HEIGHT,
            width=self.config.CHART_WIDTH,
            barmode='group',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def get_sector_colors(self) -> Dict[str, str]:
        """セクターチャートの色設定を取得"""
        return self.config.SECTOR_CHART_COLORS.copy()
    
    def set_sector_colors(self, colors: Dict[str, str]):
        """セクターチャートの色設定を更新"""
        self.config.SECTOR_CHART_COLORS.update(colors)
        self.logger.info(f"Updated sector chart colors: {colors}")
    
    def get_sector_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """セクターデータの統計情報を取得"""
        
        values = []
        for value in data.values():
            if isinstance(value, (int, float)) and not np.isnan(value):
                values.append(value)
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "positive_count": sum(1 for v in values if v > 0),
            "negative_count": sum(1 for v in values if v < 0),
            "neutral_count": sum(1 for v in values if v == 0)
        }


# ファクトリーに登録
from .base_chart_generator import ChartGeneratorFactory
ChartGeneratorFactory.register_generator("sector", SectorChartGenerator)