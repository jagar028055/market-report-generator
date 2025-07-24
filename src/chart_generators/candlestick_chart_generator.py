"""
キャンドルスティックチャート専用生成クラス
"""

import mplfinance as mpf
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt

from .base_chart_generator import BaseChartGenerator
from ..utils.exceptions import ChartGenerationError
from ..utils.error_handler import with_error_handling


class CandlestickChartGenerator(BaseChartGenerator):
    """キャンドルスティックチャート専用生成クラス"""
    
    def __init__(self, charts_dir: str = "charts", logger: Optional[Any] = None):
        super().__init__(charts_dir, logger)
        self.logger.info("Initialized CandlestickChartGenerator")
    
    @with_error_handling()
    def generate_chart(self, data: pd.DataFrame, title: str, filename: str, **kwargs) -> Optional[str]:
        """基本的なキャンドルスティックチャートを生成"""
        chart_type = kwargs.get('chart_type', 'interactive')
        
        if chart_type == 'interactive':
            return self.generate_interactive_chart(data, title, filename, **kwargs)
        else:
            return self.generate_static_chart(data, title, filename, **kwargs)
    
    def generate_interactive_chart(
        self, 
        data: pd.DataFrame, 
        title: str, 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """インタラクティブなキャンドルスティックチャートを生成"""
        
        if not self.validate_data(data, "candlestick"):
            return None
        
        try:
            # 移動平均の設定
            ma_keys = kwargs.get('ma_keys', None)
            ma_type = kwargs.get('ma_type', None)
            
            # 移動平均を計算（長期チャートの場合、デフォルト設定を使用）
            if ma_keys is not None or 'Long-Term' in title:
                data_with_ma = self._calculate_moving_averages(data, ma_keys, ma_type)
            else:
                data_with_ma = data.copy()
            
            # Plotlyキャンドルスティックチャートを作成
            fig = go.Figure(data=[go.Candlestick(
                x=data_with_ma.index,
                open=data_with_ma['Open'],
                high=data_with_ma['High'],
                low=data_with_ma['Low'],
                close=data_with_ma['Close'],
                name='Price'
            )])
            
            # 移動平均線を追加
            if ma_keys is not None or 'Long-Term' in title:
                ma_traces = self._get_ma_traces_plotly(data_with_ma, ma_keys)
                for trace in ma_traces:
                    fig.add_trace(trace)
            
            # スタイルを適用
            self._apply_plotly_style(fig, title, **kwargs)
            
                        # ファイルに保存
            output_path = self.get_output_path(filename)
            
            # リサイズ用のJavaScriptを追加
            js_resize = """<script>
    function resizePlot() {
        var chartDiv = document.getElementById('plotly-chart');
        if (chartDiv) {
            Plotly.Plots.resize(chartDiv);
        }
    }
    window.addEventListener('message', function(event) {
        if (event.data.type === 'resize-plotly') {
            resizePlot();
        }
    }, false);
    // 初期ロード時にもリサイズを試みる
    window.addEventListener('load', resizePlot);
</script>"""
            
            # HTMLを生成し、リサイズ用スクリプトを追記
            html_str = pio.to_html(fig, include_plotlyjs=self.config.PLOTLY_JS_SOURCE, full_html=True, div_id='plotly-chart')
            
            # </body>の直前にスクリプトを挿入
            html_str = html_str.replace("</body>", js_resize + "</body>")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            
            self._log_generation_result(output_path, "interactive candlestick", True)
            return str(output_path)
            
        except Exception as e:
            self._handle_generation_error(e, "interactive candlestick chart generation")
            return None
    
    def generate_static_chart(
        self, 
        data: pd.DataFrame, 
        title: str, 
        filename: str, 
        **kwargs
    ) -> Optional[str]:
        """静的なキャンドルスティックチャートを生成"""
        
        if not self.validate_data(data, "candlestick"):
            return None
        
        try:
            # 移動平均の設定
            ma_keys = kwargs.get('ma_keys', None)
            ma_type = kwargs.get('ma_type', None)
            
            # 移動平均を計算（長期チャートの場合、デフォルト設定を使用）
            if ma_keys is not None or 'Long-Term' in title:
                data_with_ma = self._calculate_moving_averages(data, ma_keys, ma_type)
            else:
                data_with_ma = data.copy()
            
            # mplfinanceスタイルを作成
            style = self._create_mplfinance_style()
            
            # 移動平均のaddplotを取得
            addplots = []
            if ma_keys is not None or 'Long-Term' in title:
                addplots = self._get_ma_addplots_mplfinance(data_with_ma, ma_keys)
            
            # チャートを生成
            fig, axlist = mpf.plot(
                data_with_ma,
                type='candle',
                style=style,
                title=title,
                ylabel='Price',
                addplot=addplots,
                returnfig=True,
                warn_too_much_data=kwargs.get('warn_too_much_data', 2000)
            )
            
            # スタイルを適用
            self.apply_chart_style(fig)
            
            # 余白を調整
            for ax in axlist:
                try:
                    ax.margins(x=0)
                except Exception:
                    pass
            fig.tight_layout()
            
            # ファイルに保存
            output_path = self.save_chart(fig, filename, 'png')
            
            # リソースをクリーンアップ
            self.cleanup_chart(fig)
            
            self._log_generation_result(output_path, "static candlestick", True)
            return output_path
            
        except Exception as e:
            self._handle_generation_error(e, "static candlestick chart generation")
            return None
    
    def generate_intraday_chart(
        self, 
        data: pd.DataFrame, 
        ticker_name: str, 
        filename: str, 
        chart_type: str = 'interactive',
        **kwargs
    ) -> Optional[str]:
        """イントラデイキャンドルスティックチャートを生成"""
        
        title = f"{ticker_name} Intraday Chart (Tokyo Time)"
        
        if chart_type == 'static':
            return self.generate_static_chart(data, title, filename, **kwargs)
        else:
            return self.generate_interactive_chart(data, title, filename, **kwargs)
    
    def generate_longterm_chart(
        self, 
        data: pd.DataFrame, 
        ticker_name: str, 
        filename: str, 
        chart_type: str = 'interactive',
        ma_keys: List[str] = None, 
        ma_type: str = None,
        **kwargs
    ) -> Optional[str]:
        """長期キャンドルスティックチャートを生成"""
        
        # 移動平均の情報をタイトルに追加
        ma_info = ""
        used_keys = ma_keys or self.config.DEFAULT_MA_DISPLAY
        ma_labels = [
            self.config.MOVING_AVERAGES[key]["label"] 
            for key in used_keys 
            if key in self.config.MOVING_AVERAGES
        ]
        if ma_labels:
            ma_type_display = ma_type or self.config.DEFAULT_MA_TYPE
            ma_info = f" ({ma_type_display}: {', '.join(ma_labels)})"
        
        title = f"{ticker_name} Long-Term Chart (1 Year){ma_info}"
        
        if chart_type == 'static':
            return self.generate_static_chart(
                data, title, filename, 
                ma_keys=ma_keys, ma_type=ma_type, 
                **kwargs
            )
        else:
            return self.generate_interactive_chart(
                data, title, filename, 
                ma_keys=ma_keys, ma_type=ma_type, 
                **kwargs
            )
    
    def _calculate_moving_averages(
        self, 
        data: pd.DataFrame, 
        ma_keys: List[str] = None, 
        ma_type: str = None
    ) -> pd.DataFrame:
        """移動平均を計算"""
        
        if ma_keys is None:
            ma_keys = self.config.DEFAULT_MA_DISPLAY
        
        if ma_type is None:
            ma_type = self.config.DEFAULT_MA_TYPE
        
        data = data.copy()
        
        for ma_key in ma_keys:
            if ma_key not in self.config.MOVING_AVERAGES:
                self.logger.warning(f"Moving average key '{ma_key}' not found in config")
                continue
            
            ma_config = self.config.MOVING_AVERAGES[ma_key]
            period = ma_config["period"]
            label = ma_config["label"]
            
            if ma_type == "SMA":
                data[label] = data['Close'].rolling(window=period).mean()
            elif ma_type == "EMA":
                data[label] = data['Close'].ewm(span=period).mean()
            elif ma_type == "WMA":
                # 重み付き移動平均の計算
                weights = pd.Series(range(1, period + 1))
                data[label] = data['Close'].rolling(window=period).apply(
                    lambda x: (x * weights).sum() / weights.sum(), raw=False
                )
            else:
                self.logger.warning(f"Unsupported MA type '{ma_type}', defaulting to SMA")
                data[label] = data['Close'].rolling(window=period).mean()
        
        return data
    
    def _get_ma_traces_plotly(self, data: pd.DataFrame, ma_keys: List[str] = None) -> List[go.Scatter]:
        """Plotly用の移動平均トレースを生成"""
        
        if ma_keys is None:
            ma_keys = self.config.DEFAULT_MA_DISPLAY
        
        traces = []
        for ma_key in ma_keys:
            if ma_key not in self.config.MOVING_AVERAGES:
                continue
            
            ma_config = self.config.MOVING_AVERAGES[ma_key]
            label = ma_config["label"]
            color = ma_config["color"]
            
            if label in data.columns:
                trace = go.Scatter(
                    x=data.index,
                    y=data[label],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=1.5)
                )
                traces.append(trace)
        
        return traces
    
    def _get_ma_addplots_mplfinance(self, data: pd.DataFrame, ma_keys: List[str] = None) -> List:
        """mplfinance用の移動平均addplotを生成"""
        
        if ma_keys is None:
            ma_keys = self.config.DEFAULT_MA_DISPLAY
        
        addplots = []
        for ma_key in ma_keys:
            if ma_key not in self.config.MOVING_AVERAGES:
                continue
            
            ma_config = self.config.MOVING_AVERAGES[ma_key]
            label = ma_config["label"]
            color = ma_config["color"]
            
            if label in data.columns:
                addplot = mpf.make_addplot(
                    data[label],
                    color=color,
                    panel=0,
                    width=0.75,
                    secondary_y=False
                )
                addplots.append(addplot)
        
        return addplots
    
    def _apply_plotly_style(self, fig: go.Figure, title: str, **kwargs):
        """Plotlyチャートにスタイルを適用"""
        
        # キャンドルスティックの色設定
        fig.update_traces(
            increasing_line_color=self.config.CANDLE_COLORS.get('up_line', 'black'),
            increasing_fillcolor=self.config.CANDLE_COLORS.get('up_fill', 'white'),
            decreasing_line_color=self.config.CANDLE_COLORS.get('down_line', 'black'),
            decreasing_fillcolor=self.config.CANDLE_COLORS.get('down_fill', 'black'),
            selector=dict(type='candlestick')
        )
        
        # レイアウトの設定
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            xaxis_title=kwargs.get('xaxis_title', 'Time'),
            yaxis_title=kwargs.get('yaxis_title', 'Price'),
            template=self.config.PLOTLY_TEMPLATE,
            height=self.config.CHART_HEIGHT,
            width=self.config.CHART_WIDTH,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    def _create_mplfinance_style(self):
        """mplfinanceスタイルを作成"""
        
        mc = mpf.make_marketcolors(
            up=self.config.CANDLE_COLORS.get('up_fill', 'white'),
            down=self.config.CANDLE_COLORS.get('down_fill', 'black'),
            edge=self.config.CANDLE_COLORS.get('up_line', 'black'),
            wick=self.config.CANDLE_COLORS.get('wick', 'black'),
            ohlc=self.config.CANDLE_COLORS.get('down_line', 'black')
        )
        
        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            rc={
                'figure.figsize': self.config.MATPLOTLIB_FIGURE_SIZE,
                'font.family': plt.rcParams['font.family']
            }
        )
        
        return style
    
    def validate_candlestick_data(self, data: pd.DataFrame) -> bool:
        """キャンドルスティックデータの特別な検証"""
        
        if not super().validate_data(data, "candlestick"):
            return False
        
        # 必要な列の存在確認
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for candlestick chart: {missing_columns}")
            return False
        
        # データの論理的整合性チェック
        invalid_rows = data[
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])
        ]
        
        if not invalid_rows.empty:
            self.logger.warning(f"Found {len(invalid_rows)} rows with invalid OHLC data")
            # 無効な行を削除
            data = data.drop(invalid_rows.index)
        
        return True
    
    def get_moving_average_config(self, ma_key: str) -> Dict[str, Any]:
        """移動平均の設定を取得"""
        return self.config.get_moving_average_config(ma_key)
    
    def add_moving_average(self, key: str, period: int, color: str, label: str):
        """移動平均設定を追加"""
        self.config.add_moving_average(key, period, color, label)
        self.logger.info(f"Added moving average: {key} = {label} ({period})")
    
    def remove_moving_average(self, key: str):
        """移動平均設定を削除"""
        self.config.remove_moving_average(key)
        self.logger.info(f"Removed moving average: {key}")


# ファクトリーに登録
from .base_chart_generator import ChartGeneratorFactory
ChartGeneratorFactory.register_generator("candlestick", CandlestickChartGenerator)