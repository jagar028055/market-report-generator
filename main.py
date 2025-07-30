import traceback
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Import all original modules to ensure all code paths are tested
from src.core.data_fetcher import DataFetcher
from src.core.chart_generator import ChartGenerator
from src.core.commentary_generator import CommentaryGenerator
from src.core.html_generator import HTMLGenerator

# Import new refactored modules
from src.async_processors.async_data_fetcher import AsyncDataFetcher
from src.async_processors.async_chart_generator import AsyncChartGenerator
from src.async_processors.async_report_generator import AsyncReportGenerator
from src.async_processors.task_manager import TaskManager, TaskPriority
from src.utils.exceptions import MarketReportException
from src.utils.error_handler import ErrorHandler
from src.config import get_system_config
import logging

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
    # Reutersからのニュース取得
    reuters_articles = fetcher.scrape_reuters_news(hours_limit=24, **reuters_config)
    
    # Google Docsからのニュース取得
    google_docs_articles = []
    google_doc_id = os.getenv("GOOGLE_DOCS_ID")
    if google_doc_id:
        try:
            google_docs_articles = fetcher.get_google_docs_news(document_id=google_doc_id, hours_limit=24)
            print(f"Google Docsから {len(google_docs_articles)} 件の記事を取得しました")
        except Exception as e:
            print(f"Google Docsからのニュース取得エラー: {e}")
    
    # 記事を統合
    news_articles = reuters_articles + google_docs_articles
    
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
    static_chart_paths = {}
    
    for name, data_set in chart_data.items():
        # イントラデイチャート（HTML版とPNG版）
        intraday_filename_html = f"{name.replace(' ', '_')}_intraday.html"
        intraday_filename_png = f"{name.replace(' ', '_')}_intraday.png"
        
        if not data_set["intraday"].empty:
            # HTMLチャート生成
            intraday_path = chart_gen.generate_intraday_chart_interactive(data_set["intraday"], name, intraday_filename_html)
            if intraday_path:
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Intraday"].append({"id": f"{sanitized_name}-intraday", "name": name, "path": f"charts/{os.path.basename(intraday_path)}", "interactive": True})
            
            # PNGチャート生成
            png_path = chart_gen.generate_intraday_chart_static(data_set["intraday"], name, intraday_filename_png)
            if png_path:
                if name not in static_chart_paths:
                    static_chart_paths[name] = {}
                static_chart_paths[name]['intraday'] = f"charts/{os.path.basename(png_path)}"
        
        # 長期チャート（HTML版とPNG版）
        longterm_filename_html = f"{name.replace(' ', '_')}_longterm.html"
        longterm_filename_png = f"{name.replace(' ', '_')}_longterm.png"
        
        if not data_set["longterm"].empty:
            # HTMLチャート生成
            longterm_path = chart_gen.generate_longterm_chart_interactive(data_set["longterm"], name, longterm_filename_html)
            if longterm_path:
                sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                grouped_charts["Long-Term"].append({"id": f"{sanitized_name}-longterm", "name": name, "path": f"charts/{os.path.basename(longterm_path)}", "interactive": True})
            
            # PNGチャート生成
            png_path = chart_gen.generate_longterm_chart_static(data_set["longterm"], name, longterm_filename_png)
            if png_path:
                if name not in static_chart_paths:
                    static_chart_paths[name] = {}
                static_chart_paths[name]['longterm'] = f"charts/{os.path.basename(png_path)}"
    
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
        sector_chart_path=f"charts/{os.path.basename(sector_chart_path)}" if sector_chart_path else None,
        static_chart_paths=static_chart_paths
    )
    print("HTMLレポート生成完了。")
    print(f"レポートは '{report_filepath}' に出力されました。")

async def async_main():
    """新しいアーキテクチャを使用した非同期メイン関数"""
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        load_dotenv()
        print("--- 米国マーケットレポート生成開始 (非同期版) ---")
        
        # TaskManagerを初期化
        task_manager = TaskManager(logger)
        
        try:
            # Phase 1: データ取得タスクを追加
            print("\n[1/4] データ取得タスクを追加中...")
            reuters_config = {
                "query": "米国市場 OR 金融 OR 経済 OR 株価 OR FRB OR FOMC OR 決算 OR 利上げ OR インフレ",
                "target_categories": ["ビジネスcategory", "マーケットcategory", "トップニュースcategory", "テクノロジーcategory", "アジア市場category", "不明", "ワールドcategory","経済category"],
                "exclude_keywords": ["スポーツ", "エンタメ", "五輪", "サッカー", "映画", "将棋", "囲碁", "芸能", "ライフ", "アングル："],
                "max_pages": 5,
                "hours_limit": 24
            }
            
            data_task_id = await task_manager.add_data_fetch_task(
                "fetch_all_market_data",
                priority=TaskPriority.HIGH,
                timeout=300.0,
                reuters_config=reuters_config
            )
            
            # Phase 2: チャートデータ取得タスクを追加（データ取得後）
            print("\n[2/4] チャートデータ取得タスクを追加中...")
            chart_data_task_id = await task_manager.add_task(
                "fetch_chart_data",
                lambda: task_manager.data_fetcher.fetch_chart_data_async(),
                priority=TaskPriority.MEDIUM,
                timeout=300.0,
                dependencies=[data_task_id]
            )
            
            # Phase 3: チャート生成タスクを追加（チャートデータ取得後）
            print("\n[3/4] チャート生成タスクを追加中...")
            chart_gen_task_id = await task_manager.add_task(
                "generate_charts",
                lambda: task_manager.chart_generator.generate_all_charts(task_manager.get_task_result(chart_data_task_id).result),
                priority=TaskPriority.MEDIUM,
                timeout=300.0,
                dependencies=[chart_data_task_id]
            )
            
            # Phase 4: レポート生成タスクを追加（すべて完了後）
            print("\n[4/4] レポート生成タスクを追加中...")
            report_task_id = await task_manager.add_report_generation_task(
                "generate_final_report",
                priority=TaskPriority.LOW,
                timeout=300.0,
                dependencies=[data_task_id, chart_gen_task_id]
            )
            
            # すべてのタスクを実行
            print("\n--- タスク実行開始 ---")
            results = await task_manager.execute_all_tasks()
            
            # 結果を確認
            print("\n--- タスク実行完了 ---")
            stats = task_manager.get_statistics()
            print(f"実行統計: {stats}")
            
            # 成功したタスクの結果を表示
            successful_tasks = [
                task_id for task_id, result in results.items() 
                if result.status.name == 'COMPLETED'
            ]
            print(f"成功したタスク: {len(successful_tasks)}/{len(results)}")
            
            # 最終レポートの場所を出力
            if report_task_id in results and results[report_task_id].result:
                print(f"レポートが生成されました: {results[report_task_id].result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Async execution failed: {e}")
            raise
        
        finally:
            # リソースをクリーンアップ
            task_manager.cleanup()
            
    except Exception as e:
        logger.error(f"Async main failed: {e}")
        raise


def enhanced_main():
    """改良された同期メイン関数（新しいクラスを使用）"""
    
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        load_dotenv()
        print("--- 米国マーケットレポート生成開始 (改良版) ---")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 新しいクラスを使用してデータ取得
        print("\n[1/5] データ取得中...")
        from src.data_fetchers.market_data_fetcher import MarketDataFetcher
        from src.data_fetchers.news_data_fetcher import NewsDataFetcher
        from src.data_fetchers.economic_data_fetcher import EconomicDataFetcher
        
        market_fetcher = MarketDataFetcher(logger)
        news_fetcher = NewsDataFetcher(logger)
        economic_fetcher = EconomicDataFetcher(logger)
        
        market_data = market_fetcher.get_market_data()
        economic_indicators = economic_fetcher.get_economic_indicators()
        sector_performance = market_fetcher.get_sector_etf_performance()
        
        # ロイターからのスクレイピングは維持しつつ、Google Docsからの取得に切り替え
        # reuters_config = {
        #     "query": "米国市場 OR 金融 OR 経済 OR 株価 OR FRB OR FOMC OR 決算 OR 利上げ OR インフレ",
        #     "target_categories": ["ビジネスcategory", "マーケットcategory", "トップニュースcategory", "ワールドcategory", "テクノロジーcategory", "アジア市場category","不明","ワールドcategory","経済category"],
        #     "exclude_keywords": ["スポーツ", "エンタメ", "五輪", "サッカー", "映画", "将棋", "囲碁", "芸能", "ライフ", "アングル："],
        #     "max_pages": 5
        # }
        # news_articles = news_fetcher.scrape_reuters_news(hours_limit=24, **reuters_config)
        
        google_doc_id = os.getenv("GOOGLE_DOCS_ID")
        if google_doc_id:
            news_articles = news_fetcher.get_google_docs_news(document_id=google_doc_id, hours_limit=24)
        else:
            print("警告: GOOGLE_DOCS_ID環境変数が設定されていません。Google Docsからのニュース取得をスキップします。")
            news_articles = []
        
        # チャートデータ取得
        chart_data = {}
        for name, ticker in market_fetcher.tickers.items():
            if name not in ["米国2年金利"]:
                chart_data[name] = {
                    "intraday": market_fetcher.get_intraday_data(ticker),
                    "longterm": market_fetcher.get_historical_data(ticker, period="1y")
                }
        print("データ取得完了。")
        
        # 新しいチャートジェネレーターを使用
        print("\n[2/5] チャート生成中...")
        charts_output_dir = os.path.join(base_dir, "charts")
        from src.chart_generators import create_chart_generator
        
        chart_gen = create_chart_generator("integrated", charts_dir=charts_output_dir)
        grouped_charts = {"Intraday": [], "Long-Term": []}
        static_chart_paths = {}
        
        for name, data_set in chart_data.items():
            # インタラクティブチャート（HTML版）
            intraday_filename = f"{name.replace(' ', '_')}_intraday.html"
            if not data_set["intraday"].empty:
                intraday_path = chart_gen.generate_intraday_chart_interactive(data_set["intraday"], name, intraday_filename)
                if intraday_path:
                    sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    grouped_charts["Intraday"].append({"id": f"{sanitized_name}-intraday", "name": name, "path": f"charts/{os.path.basename(intraday_path)}", "interactive": True})
            
            # 静的チャート（PNG版）- ポップアップ用
            intraday_static_filename = f"{name.replace(' ', '_')}_intraday_static.png"
            if not data_set["intraday"].empty:
                static_path = chart_gen.generate_intraday_chart_static(data_set["intraday"], name, intraday_static_filename)
                if static_path:
                    if name not in static_chart_paths:
                        static_chart_paths[name] = {}
                    static_chart_paths[name]['intraday'] = f"charts/{os.path.basename(static_path)}"
            
            longterm_filename = f"{name.replace(' ', '_')}_longterm.html"
            if not data_set["longterm"].empty:
                longterm_path = chart_gen.generate_longterm_chart_interactive(data_set["longterm"], name, longterm_filename)
                if longterm_path:
                    sanitized_name = name.replace(' ', '-').replace('&', 'and').replace('.', '').lower()
                    grouped_charts["Long-Term"].append({"id": f"{sanitized_name}-longterm", "name": name, "path": f"charts/{os.path.basename(longterm_path)}", "interactive": True})
            
            # 長期静的チャート（PNG版）
            longterm_static_filename = f"{name.replace(' ', '_')}_longterm_static.png"
            if not data_set["longterm"].empty:
                static_longterm_path = chart_gen.generate_longterm_chart_static(data_set["longterm"], name, longterm_static_filename)
                if static_longterm_path:
                    if name not in static_chart_paths:
                        static_chart_paths[name] = {}
                    static_chart_paths[name]['longterm'] = f"charts/{os.path.basename(static_longterm_path)}"
        print("チャート生成完了。")
        
        # AIコメント生成（従来と同じ）
        print("\n[3/5] AIコメント生成中...")
        comment_gen = CommentaryGenerator()
        commentary = comment_gen.generate_market_commentary(news_articles, economic_indicators)
        print("AIコメント生成完了。")
        
        # セクター別ETFチャート生成
        print("\n[4/5] セクター別ETFチャート生成中...")
        if sector_performance:
            sector_performance_sorted = dict(sorted(sector_performance.items(), key=lambda item: item[1] if item[1] is not None else -float('inf'), reverse=True))
            sector_chart_path = chart_gen.generate_sector_performance_chart(sector_performance_sorted, "sector_performance_chart.html")
            print("セクター別ETFチャート生成完了。")
        else:
            sector_chart_path = None
            print("セクターデータが取得できませんでした。")
        
        # HTMLレポート生成（従来と同じ）
        print("\n[5/5] HTMLレポート生成中...")
        html_gen = HTMLGenerator(output_dir=base_dir, report_filename="index.html")
        report_filepath = html_gen.generate_report(
            market_data=market_data, economic_indicators=economic_indicators, sector_performance=sector_performance,
            news_articles=news_articles, commentary=commentary, grouped_charts=grouped_charts,
            sector_chart_path=f"charts/{os.path.basename(sector_chart_path)}" if sector_chart_path else None,
            static_chart_paths=static_chart_paths
        )
        print("HTMLレポート生成完了。")
        print(f"レポートは '{report_filepath}' に出力されました。")
        
        # クリーンアップ
        market_fetcher.cleanup()
        news_fetcher.cleanup()
        economic_fetcher.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced main failed: {e}")
        raise


if __name__ == "__main__":
    # 実行モードの選択
    import sys
    
    # コマンドライン引数で実行モードを選択
    mode = sys.argv[1] if len(sys.argv) > 1 else "enhanced"
    
    try:
        if mode == "async":
            print("非同期モードで実行中...")
            asyncio.run(async_main())
        elif mode == "enhanced":
            print("改良版モードで実行中...")
            enhanced_main()
        elif mode == "original":
            print("従来版モードで実行中...")
            original_main()
        else:
            print("有効なモード: async, enhanced, original")
            sys.exit(1)
        
        with open("success.log", "w") as f:
            f.write(f"Script completed successfully in {mode} mode.")
        
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"An error occurred:\n{error_message}")
        with open("error.log", "w") as f:
            f.write(f"Error in {mode} mode:\n{error_message}")
        # Re-raise the exception to ensure the process exits with an error code
        raise
