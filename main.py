from data_fetcher import DataFetcher
from chart_generator import ChartGenerator
from commentary_generator import CommentaryGenerator
from html_generator import HTMLGenerator
import os
from datetime import datetime
from dotenv import load_dotenv

def main():
    load_dotenv() # .envファイルを読み込む
    print("--- 米国マーケットレポート生成開始 ---")

    # 1. データ取得
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
    news_articles = fetcher.scrape_reuters_news(
        hours_limit=24, # 週末でも記事を取得できるよう72時間に延長
        **reuters_config
    )
    
    # チャート生成用のデータも取得
    chart_data = {}
    for name, ticker in fetcher.tickers.items():
        if name not in ["米国2年金利"]: # 金利はyfinanceでチャートデータが取得しにくい場合があるため除外
            chart_data[name] = {
                "intraday": fetcher.get_intraday_data(ticker), # days引数を削除
                "longterm": fetcher.get_historical_data(ticker, period="1y")
            }
    print("データ取得完了。")

    # 2. チャート生成
    print("\n[2/5] チャート生成中...")
    # 絶対パスで指定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    charts_output_dir = os.path.join(base_dir, "charts")
    chart_gen = ChartGenerator(charts_dir=charts_output_dir)
    
    grouped_charts = {
        "Intraday": [],
        "Long-Term": []
    }
    for name, data_set in chart_data.items():
        # イントラデイチャート
        intraday_filename = f"{name.replace(' ', '_')}_intraday.html"
        if not data_set["intraday"].empty:
            intraday_path = chart_gen.generate_intraday_chart_interactive(data_set["intraday"], name, intraday_filename)
            if intraday_path:
                # idをサニタイズ
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Intraday"].append({
                    "id": f"{sanitized_name}-intraday",
                    "name": name,
                    "path": f"charts/{os.path.basename(intraday_path)}", # HTMLからの相対パス
                    "interactive": True
                })
            else:
                print(f"Warning: Intraday chart for {name} could not be generated.")
        else:
            print(f"Skipping intraday chart for {name} due to empty data.")

        # 長期チャート
        longterm_filename = f"{name.replace(' ', '_')}_longterm.html"
        if not data_set["longterm"].empty:
            longterm_path = chart_gen.generate_longterm_chart_interactive(data_set["longterm"], name, longterm_filename)
            if longterm_path:
                # idをサニタイズ
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Long-Term"].append({
                    "id": f"{sanitized_name}-longterm",
                    "name": name,
                    "path": f"charts/{os.path.basename(longterm_path)}", # HTMLからの相対パス
                    "interactive": True
                })
            else:
                print(f"Warning: Long-term chart for {name} could not be generated.")
        else:
            print(f"Skipping long-term chart for {name} due to empty data.")
    print("チャート生成完了。")

    # 3. AIコメント生成
    print("\n[3/5] AIコメント生成中...")
    try:
        comment_gen = CommentaryGenerator()
        commentary = comment_gen.generate_market_commentary(news_articles, economic_indicators)
    except ValueError as e:
        print(f"AIコメント生成エラー: {e}")
        commentary = "AIコメントの生成に失敗しました。GEMINI_API_KEYが正しく設定されているか確認してください。"
    print("AIコメント生成完了。")

    # 4. セクター別ETFチャート生成
    print("\n[4/5] セクター別ETFチャート生成中...")
    sector_performance_sorted = dict(sorted(sector_performance.items(), key=lambda item: item[1] if item[1] is not None else -float('inf'), reverse=True))
    sector_chart_path = chart_gen.generate_sector_performance_chart(sector_performance_sorted, "sector_performance_chart.html")
    print("セクター別ETFチャート生成完了。")

    # 5. HTMLレポート生成
    print("\n[5/5] HTMLレポート生成中...")
    try:
        # 絶対パスで指定
        html_gen = HTMLGenerator(
            template_dir=os.path.join(base_dir, "templates"),
            output_dir=base_dir
        )
        report_filepath = html_gen.generate_report(
            market_data=market_data,
            economic_indicators=economic_indicators,
            sector_performance=sector_performance,
            news_articles=news_articles,
            commentary=commentary,
            grouped_charts=grouped_charts, # タイムスタンプ付きのパスはhtml_generatorで付与される
            sector_chart_path=f"charts/{os.path.basename(sector_chart_path)}" if sector_chart_path else None
        )
        print("HTMLレポート生成完了。")
    except Exception as e:
        import traceback
        print(f"--- HTMLレポート生成中にエラーが発生しました ---")
        print(f"エラータイプ: {type(e).__name__}")
        print(f"エラーメッセージ: {e}")
        print("--- トレースバック ---")
        traceback.print_exc()
        print("--------------------")
        # エラーが発生したことを示すために、Noneを返すか、あるいは例外を再送出する
        # ここではプロセスを終了させるために例外を再送出します
        raise

    print("\n--- 米国マーケットレポート生成完了 ---")
    print(f"レポートは '{report_filepath}' に出力されました。")

if __name__ == "__main__":
    main()
