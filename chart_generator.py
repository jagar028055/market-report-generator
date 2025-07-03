import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.io as pio

class ChartGenerator:
    def __init__(self, charts_dir="market_report_generator/charts"):
        self.charts_dir = charts_dir
        os.makedirs(self.charts_dir, exist_ok=True)
        self._setup_japanese_font()

    def _setup_japanese_font(self):
        """matplotlibで日本語を表示するためのフォント設定"""
        # macOSの場合の一般的な日本語フォント (優先順位を調整)
        font_paths = [
            '/System/Library/Fonts/ヒラギノ角ゴ ProN W3.ttc', # ヒラギノ角ゴ ProN (優先)
            '/System/Library/Fonts/Hiragino Sans/Hiragino Sans W3.ttc', # macOS Sonoma以降のパス
            '/System/Library/Fonts/Supplemental/ヒラギノ角ゴ ProN W3.ttc', # 旧パス
            '/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/Osaka.ttf',
            '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
            '/System/Library/Fonts/ヒラギノ明朝 ProN W3.ttc',
        ]
        
        self.font_paths = font_paths # font_pathsをインスタンス変数として保存 (デバッグ用)
        
        self.japanese_font_path = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                self.japanese_font_path = font_path
                break
        
        if self.japanese_font_path:
            font_name = fm.FontProperties(fname=self.japanese_font_path).get_name()
            plt.rcParams['font.family'] = [font_name] + plt.rcParams['font.family'] # 先頭に追加して優先
            plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止
            print(f"Matplotlib: 日本語フォント '{font_name}' を設定しました。")
        else:
            print("Warning: 日本語フォントが見つかりませんでした。グラフの日本語が正しく表示されない可能性があります。")
            print("Please install a Japanese font or specify its path manually.")

    def _get_japanese_font_path(self):
        """設定されている日本語フォントのパスを返すヘルパー関数"""
        return self.japanese_font_path

    def generate_intraday_chart_interactive(self, data: pd.DataFrame, ticker_name: str, filename: str):
        """
        イントラデイチャートを Plotly でインタラクティブ表示用 HTML として保存する
        """
        if data.empty:
            print(f"No data to generate interactive intraday chart for {ticker_name}.")
            return None
        filepath = os.path.join(self.charts_dir, filename)
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        # 白黒ベースの配色に変更
        fig.update_traces(
            increasing_line_color='black',
            increasing_fillcolor='white',
            decreasing_line_color='black',
            decreasing_fillcolor='black',
            selector=dict(type='candlestick')
        )
        # レンジスライダー（下段ミニチャート）を非表示に
        fig.update_layout(title=f"{ticker_name} Intraday Chart (Tokyo Time)",
                          xaxis_rangeslider_visible=False,
                          xaxis_title='Time',
                          yaxis_title='Price',
                          template='simple_white',
                          height=600)
        pio.write_html(fig, file=filepath, include_plotlyjs='cdn', full_html=True)
        print(f"Interactive intraday chart saved to {filepath}")
        return filepath

    def generate_longterm_chart_interactive(self, data: pd.DataFrame, ticker_name: str, filename: str):
        """
        1年ローソク足を Plotly でインタラクティブ表示用 HTML として保存する
        """
        if data.empty:
            print(f"No data to generate interactive long-term chart for {ticker_name}.")
            return None
        filepath = os.path.join(self.charts_dir, filename)
        # 移動平均
        data = data.copy()
        data['MA25'] = data['Close'].rolling(window=25).mean()
        data['MA75'] = data['Close'].rolling(window=75).mean()

        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'],
                                             name='Price')])
        fig.add_trace(go.Scatter(x=data.index, y=data['MA25'], mode='lines', name='MA25', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA75'], mode='lines', name='MA75', line=dict(color='red')))
        # 白黒ベースの配色に変更
        fig.update_traces(
            increasing_line_color='black',
            increasing_fillcolor='white',
            decreasing_line_color='black',
            decreasing_fillcolor='black',
            selector=dict(type='candlestick')
        )
        # レンジスライダーを非表示に
        fig.update_layout(title=f"{ticker_name} Long-Term Chart (1 Year)",
                          xaxis_rangeslider_visible=False,
                          xaxis_title='Date',
                          yaxis_title='Price',
                          template='simple_white',
                          height=600)
        pio.write_html(fig, file=filepath, include_plotlyjs='cdn', full_html=True)
        print(f"Interactive long-term chart saved to {filepath}")
        return filepath

    # ---------- existing static matplotlib functions below ----------
    def generate_intraday_chart(self, data: pd.DataFrame, ticker_name: str, filename: str):
        """
        イントラデイチャートを生成し、画像として保存する。
        横軸は東京時間で表示されるように、データフレームのインデックスは既に東京時間になっていることを前提とする。
        """
        if data.empty:
            print(f"No data to generate intraday chart for {ticker_name}.")
            return None

        filepath = os.path.join(self.charts_dir, filename)
        
        # mplfinanceのスタイル設定でフォントを明示的に指定
        # 白黒ローソク足のスタイルを定義
        mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black', ohlc='black')
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, rc={
            'figure.figsize':(12, 6), # ウェブ表示に適したサイズに調整
            'font.family': plt.rcParams['font.family'] # 設定した日本語フォントを使用
        })
        
        # mpf.plot 内で savefig を行うと調整が効かないため、自前で保存する
        fig, axlist = mpf.plot(data,
                               type='candle',
                               style=s,
                               title=f"{ticker_name} Intraday Chart (Tokyo Time)",
                               ylabel='Price',
                               returnfig=True,
                               warn_too_much_data=2000)

        # 左側余白を削減
        for ax in axlist:
            try:
                ax.margins(x=0)  # x 方向余白ゼロ
            except Exception:
                pass
        fig.tight_layout()

        fig.savefig(filepath, dpi=150, format='png', bbox_inches='tight')
        
        plt.close(fig) # メモリ解放
        print(f"Intraday chart saved to {filepath}")
        return filepath

    def generate_longterm_chart(self, data: pd.DataFrame, ticker_name: str, filename: str):
        """
        長期チャート（1年程度）を生成し、画像として保存する。
        移動平均線（例: 25日、75日）も表示する。
        """
        if data.empty:
            print(f"No data to generate long-term chart for {ticker_name}.")
            return None

        filepath = os.path.join(self.charts_dir, filename)
        
        data['MA25'] = data['Close'].rolling(window=25).mean()
        data['MA75'] = data['Close'].rolling(window=75).mean()

        # 白黒ローソク足のスタイルを定義
        mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black', ohlc='black')
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, rc={
            'figure.figsize':(12, 6), # ウェブ表示に適したサイズに調整
            'font.family': plt.rcParams['font.family'] # 設定した日本語フォントを使用
        })
        
        apds = [
            mpf.make_addplot(data['MA25'], color='blue', panel=0, width=0.75, secondary_y=False, ylabel='Price'),
            mpf.make_addplot(data['MA75'], color='red', panel=0, width=0.75, secondary_y=False, ylabel='Price')
        ]

        fig, axlist = mpf.plot(data,
                               type='candle',
                               style=s,
                               title=f"{ticker_name} Long-Term Chart (1 Year)",
                               ylabel='Price',
                               addplot=apds,
                               returnfig=True,
                               warn_too_much_data=500)

        for ax in axlist:
            try:
                ax.margins(x=0)
            except Exception:
                pass
        fig.tight_layout()

        fig.savefig(filepath, dpi=150, format='png', bbox_inches='tight')
        
        plt.close(fig) # メモリ解放
        print(f"Long-term chart saved to {filepath}")
        return filepath

        # 既存静的グラフはそのまま
    def generate_sector_performance_chart(self, data: dict, filename: str):
        """
        セクター別ETFの変化率を横棒グラフで生成し、SVG画像として保存する。
        変化率が高い順にソートし、正の値は緑、負の値は赤で表示する。
        """
        if not data:
            print("No data to generate sector performance chart.")
            return None

        filepath = os.path.join(self.charts_dir, filename)

        # データを変化率でソート (降順)
        sorted_sectors = sorted(data.items(), key=lambda item: item[1], reverse=True)
        
        # セクター名を短縮
        sectors = []
        for name, _ in sorted_sectors:
            short_name = name.replace(" Select Sector SPDR Fund", "").replace(" Select Sector PDR Fund", "").strip()
            sectors.append(short_name)
            
        performance = [item[1] for item in sorted_sectors]

        # Plotly インタラクティブ横棒グラフ
        colors = ["green" if p > 0 else "red" for p in performance]
        fig = go.Figure(go.Bar(
            x=performance,
            y=sectors,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.2f}%" for p in performance],
            textposition='auto'
        ))
        fig.update_layout(
            template='simple_white',
            title='米国セクターETF変化率',
            xaxis_title='変化率 (%)',
            yaxis=dict(autorange='reversed'),
            height=max(400, len(sectors)*40 + 100)
        )
        pio.write_html(fig, file=filepath, include_plotlyjs='cdn', full_html=True)
        print(f"Interactive sector performance chart saved to {filepath}")
        return filepath

def _get_japanese_font_path(self):
    """設定されている日本語フォントのパスを返すヘルパー関数"""
    return self.japanese_font_path

if __name__ == "__main__":
    # テスト用のダミーデータ生成
    # 実際のデータはdata_fetcherから取得する
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'Open': 100 + (pd.np.random.rand(200) - 0.5) * 10,
        'High': 105 + (pd.np.random.rand(200) - 0.5) * 10,
        'Low': 95 + (pd.np.random.rand(200) - 0.5) * 10,
        'Close': 100 + (pd.np.random.rand(200) - 0.5) * 10,
        'Volume': pd.np.random.randint(100000, 500000, 200)
    }, index=dates)
    data.index.name = 'Date'

    # イントラデイ用のダミーデータ（東京時間）
    intraday_dates = pd.date_range(start='2025-06-15 09:00', periods=100, freq='5min', tz='Asia/Tokyo')
    intraday_data = pd.DataFrame({
        'Open': 200 + (pd.np.random.rand(100) - 0.5) * 5,
        'High': 202 + (pd.np.random.rand(100) - 0.5) * 5,
        'Low': 198 + (pd.np.random.rand(100) - 0.5) * 5,
        'Close': 200 + (pd.np.random.rand(100) - 0.5) * 5,
        'Volume': pd.np.random.randint(1000, 5000, 100)
    }, index=intraday_dates)
    intraday_data.index.name = 'Date'

    # セクターパフォーマンスのダミーデータ
    dummy_sector_perf = {
        "Technology": 1.20,
        "Financial": -0.50,
        "Health Care": 0.80,
        "Energy": 2.50,
        "Utilities": -0.10
    }

    chart_gen = ChartGenerator()
    chart_gen.generate_longterm_chart(data, "Dummy Stock", "dummy_longterm.svg")
    chart_gen.generate_intraday_chart(intraday_data, "Dummy Stock", "dummy_intraday.svg")
    chart_gen.generate_sector_performance_chart(dummy_sector_perf, "dummy_sector_performance.svg")
