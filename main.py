import traceback

# Import all original modules to ensure all code paths are tested
from src.core.data_fetcher import DataFetcher
from src.core.chart_generator import ChartGenerator
from src.core.commentary_generator import CommentaryGenerator
from src.core.html_generator import HTMLGenerator
import os
from datetime import datetime
from dotenv import load_dotenv

def original_main():
    # This is the original main function.
    # We will call this inside a try-except block.
    load_dotenv()
    print("--- 米国マーケットレポート生成開始 ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/5] データ取得中...")
    fetcher = DataFetcher()
    market_data = fetcher.get_market_data()
    economic_indicators = fetcher.get_economic_indicators()
    sector_performance = fetcher.get_sector_etf_performance()
    reuters_config = {
        "query": "米国市場 OR 金融 OR 経済 OR 株価 OR FRB OR FOMC OR 決算 OR 利上げ OR インフレ",
        "target_categories": ["ビジネスcategory", "マーケットcategory", "トップニュースcategory", "ワールドcategory", "テクノロジーcategory", "アジア市場category","不明","ワールドcategory","経済category"],
        "exclude_keywords": ["スポーツ", "エンタメ", "五輪", "サッカー", "映画", "将棋", "囲碁", "芸能", "ライフ", "アングル："],
        "max_pages": 5
    }
    news_articles = fetcher.scrape_reuters_news(hours_limit=24, **reuters_config)
    
    chart_data = {}
    for name, ticker in fetcher.tickers.items():
        if name not in ["米国2年金利"]:
            chart_data[name] = {
                "intraday": fetcher.get_intraday_data(ticker),
                "longterm": fetcher.get_historical_data(ticker, period="1y")
            }
    print("データ取得完了。")

    print("\n[2/5] チャート生成中...")
    charts_output_dir = os.path.join(base_dir, "charts")
    chart_gen = ChartGenerator(charts_dir=charts_output_dir)
    grouped_charts = {"Intraday": [], "Long-Term": []}
    for name, data_set in chart_data.items():
        intraday_filename = f"{name.replace(' ', '_')}_intraday.html"
        if not data_set["intraday"].empty:
            intraday_path = chart_gen.generate_intraday_chart_interactive(data_set["intraday"], name, intraday_filename)
            if intraday_path:
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Intraday"].append({"id": f"{sanitized_name}-intraday", "name": name, "path": f"charts/{os.path.basename(intraday_path)}", "interactive": True})
        longterm_filename = f"{name.replace(' ', '_')}_longterm.html"
        if not data_set["longterm"].empty:
            longterm_path = chart_gen.generate_longterm_chart_interactive(data_set["longterm"], name, longterm_filename)
            if longterm_path:
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Long-Term"].append({"id": f"{sanitized_name}-longterm", "name": name, "path": f"charts/{os.path.basename(longterm_path)}", "interactive": True})
    print("チャート生成完了。")

    print("\n[3/5] AIコメント生成中...")
    comment_gen = CommentaryGenerator()
    commentary = comment_gen.generate_market_commentary(news_articles, economic_indicators)
    print("AIコメント生成完了。")

    print("\n[4/5] セクター別ETFチャート生成中...")
    sector_performance_sorted = dict(sorted(sector_performance.items(), key=lambda item: item[1] if item[1] is not None else -float('inf'), reverse=True))
    sector_chart_path = chart_gen.generate_sector_performance_chart(sector_performance_sorted, "sector_performance_chart.html")
    print("セクター別ETFチャート生成完了。")

    print("\n[5/5] HTMLレポート生成中...")
    html_gen = HTMLGenerator(output_dir=base_dir, report_filename="index.html")
    report_filepath = html_gen.generate_report(
        market_data=market_data, economic_indicators=economic_indicators, sector_performance=sector_performance,
        news_articles=news_articles, commentary=commentary, grouped_charts=grouped_charts,
        sector_chart_path=f"charts/{os.path.basename(sector_chart_path)}" if sector_chart_path else None
    )
    print("HTMLレポート生成完了。")
    print(f"レポートは '{report_filepath}' に出力されました。")

if __name__ == "__main__":
    try:
        original_main()
        with open("success.log", "w") as f:
            f.write("Script completed successfully.")
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"An error occurred:\n{error_message}")
        with open("error.log", "w") as f:
            f.write(error_message)
        # Re-raise the exception to ensure the process exits with an error code
        raise
