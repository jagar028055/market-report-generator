from jinja2 import Environment, FileSystemLoader
import markdown
import os
from datetime import datetime

class HTMLGenerator:
    def __init__(self, output_dir=".", report_filename="market_report.html"):
        # このファイルの場所を基準にtemplatesディレクトリへの絶対パスを構築
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.join(base_dir, 'templates')
        self.report_filename = report_filename
        self.output_dir = output_dir
        # FileSystemLoaderには絶対パスを渡す
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def generate_report(self, market_data: dict, economic_indicators: dict, sector_performance: dict, news_articles: list, commentary: str, grouped_charts: dict, sector_chart_path: str = None):
        """
        収集したデータをHTMLレポートとして生成する。
        """
        # __init__でenvが初期化されているので、再初期化は不要
        template = self.env.get_template('report_template.html')
        
        report_date = datetime.now().strftime("%Y年%m月%d日")
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

        # CSSファイルのパスをHTMLファイルからの相対パスとして生成
        css_path = "static/style.css" # market_report.htmlと同じディレクトリからの相対パス

        # grouped_charts内の各チャートパスにキャッシュバスティング用のタイムスタンプを追加
        # これにより、ブラウザが常に最新の画像を読み込むように強制する
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for chart_type, charts_list in grouped_charts.items():
            for chart in charts_list:
                chart['path_with_buster'] = f"{chart['path']}?v={current_timestamp}"
        
        # sector_chart_pathにもキャッシュバスティング用のタイムスタンプを追加
        if sector_chart_path:
            sector_chart_path_with_buster = f"{sector_chart_path}?v={current_timestamp}"
        else:
            sector_chart_path_with_buster = None

                # MarkdownコメントをHTMLへ変換
        commentary_html = markdown.markdown(commentary, extensions=[
            'extra',        # tables, etc.
            'nl2br',        # 改行を<br>に
            'sane_lists'
        ])

        output_html = template.render(
            report_date=report_date,
            generation_time=generation_time,
            market_data=market_data,
            economic_indicators=economic_indicators,
            sector_performance=sector_performance,
            news_articles=news_articles,
            commentary=commentary_html,
            grouped_charts=grouped_charts, # タイムスタンプ付きのチャートデータ構造を渡す
            sector_chart_path=sector_chart_path_with_buster, # タイムスタンプ付きのセクターチャートパスをテンプレートに渡す
            css_path=css_path # CSSパスをテンプレートに渡す
        )

        output_filepath = os.path.join(self.output_dir, self.report_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(output_html)
        print(f"HTML report generated at {output_filepath}")
        return output_filepath

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_market_data = {
        "S&P500": {"current": "5200.50", "change": "+25.30", "change_percent": "+0.49%"},
        "NASDAQ100": {"current": "18000.75", "change": "+100.20", "change_percent": "+0.56%"},
        "米国10年金利": {"current": "4.250", "change": "+0.050", "change_bp": "+5.00", "change_percent": "+1.19%"}
    }
    dummy_economic_indicators = {
        "yesterday": [
            {"name": "CPI", "previous": "3.0%", "actual": "3.2%", "forecast": "3.1%"}
        ],
        "today_scheduled": [
            {"name": "PMI", "previous": "52.0", "forecast": "52.5"}
        ]
    }
    dummy_sector_performance = {
        "Technology Select Sector SPDR Fund": "+1.20%",
        "Financial Select Sector SPDR Fund": "+0.50%"
    }
    dummy_news = [
        {"title": "Tech stocks rally on strong earnings", "link": "#", "published_jst": datetime.now()},
        {"title": "Fed hints at rate cut", "link": "#", "published_jst": datetime.now()}
    ]
    dummy_commentary = """
    株式市場: 前日の米国株式市場は、テクノロジー株がけん引し上昇しました。特に、主要企業の好決算が発表されたことで、投資家心理が改善しました。
    金利市場: 金利市場では、FRB議長の発言を受けて長期金利が上昇しました。インフレ抑制への強い姿勢が示されたことが背景にあります。
    為替市場: 為替市場では、ドルが主要通貨に対して堅調に推移しました。経済指標の発表もドル高を後押ししました。
    """
    # 新しいダミーチャートデータ構造
    dummy_grouped_charts = {
        "Intraday": [
            {"id": "sp500-intraday", "name": "S&P 500", "path": "charts/sp500_intraday.png"},
            {"id": "nasdaq-intraday", "name": "NASDAQ 100", "path": "charts/nasdaq_intraday.png"}
        ],
        "Long-Term": [
            {"id": "sp500-longterm", "name": "S&P 500", "path": "charts/sp500_longterm.png"},
            {"id": "nasdaq-longterm", "name": "NASDAQ 100", "path": "charts/nasdaq_longterm.png"}
        ]
    }

    html_gen = HTMLGenerator(
        template_dir="templates",
        output_dir="."
    )
    html_gen.generate_report(
        market_data=dummy_market_data,
        economic_indicators=dummy_economic_indicators,
        sector_performance=dummy_sector_performance,
        news_articles=dummy_news,
        commentary=dummy_commentary,
        grouped_charts=dummy_grouped_charts # 新しいチャートデータ構造を渡す
    )
