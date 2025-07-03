import yfinance as yf
import investpy
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys
import time
import re
import numpy as np # numpyをインポート
import os
import google.generativeai as genai
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

class DataFetcher:
    def __init__(self):
        self.tickers = {
            "S&P500": "^GSPC",
            "NASDAQ100": "^NDX",
            "ダウ30": "^DJI",
            "SOX": "^SOX",
            "米国2年金利": "^TNX", 
            "米国10年金利": "^TNX",
            "ドル円": "JPY=X",
            "ユーロドル": "EURUSD=X",
            "ビットコイン": "BTC-USD",
            "ゴールド": "GC=F",
            "原油": "CL=F",
            "VIX": "^VIX"
        }
        self.sector_etfs = {
            "XLK": "Technology Select Sector SPDR Fund",
            "XLF": "Financial Select Sector SPDR Fund",
            "XLV": "Health Care Select Sector SPDR Fund",
            "XLI": "Industrial Select Sector SPDR Fund",
            "XLY": "Consumer Discretionary Select Sector SPDR Fund",
            "XLP": "Consumer Staples Select Sector PDR Fund",
            "XLE": "Energy Select Sector SPDR Fund",
            "XLU": "Utilities Select Sector SPDR Fund",
            "XLB": "Materials Select Sector SPDR Fund",
            "XLRE": "Real Estate Select Sector SPDR Fund",
            "XLC": "Communication Services Select Sector SPDR Fund"
        }
        
        # 米経済指標名の英→日本語変換マップ
        self.indicator_translations = {
            "Initial Jobless Claims": "新規失業保険申請件数",
            "GDP (QoQ)": "実質GDP（前期比年率）",
            "GDP Price Index (QoQ)": "GDP価格指数（前期比）",
            "Core PCE Price Index (MoM)": "コアPCE価格指数（前月比）",
            "Core PCE Price Index (YoY)": "コアPCE価格指数（前年比）",
            "PCE Price Index (MoM)": "PCE価格指数（前月比）",
            "PCE Price Index (YoY)": "PCE価格指数（前年比）",
            "Existing Home Sales": "中古住宅販売件数",
            "New Home Sales": "新築住宅販売件数",
            "Consumer Confidence": "消費者信頼感指数",
            "ISM Manufacturing PMI": "ISM製造業景況指数",
            "ISM Non-Manufacturing PMI": "ISM非製造業景況指数",
            "Retail Sales (MoM)": "小売売上高（前月比）",
            "CPI (MoM)": "消費者物価指数（前月比）",
            "CPI (YoY)": "消費者物価指数（前年比）",
            "Core CPI (MoM)": "コア消費者物価指数（前月比）",
            "Core CPI (YoY)": "コア消費者物価指数（前年比）",
            "Unemployment Rate": "失業率"
        }

        self.ASSET_CLASSES = {
            "US_STOCK": ["^GSPC", "^DJI", "^NDX", "^SOX", "^TNX", "^VIX"],
            "24H_ASSET": ["JPY=X", "EURUSD=X", "BTC-USD", "GC=F", "CL=F"]
        }
        self.INTRADAY_INTERVAL = "5m"
        self.INTRADAY_PERIOD_DAYS = 7

        # 未訳の経済指標を保存するファイル
        self.untranslated_file = os.path.join(os.path.dirname(__file__), "untranslated_indicators.txt")

        # Selenium WebDriverのオプション設定
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920x1080")
        self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36')
        
        # ChromeDriverのパスをシステムパスに追加 (Homebrewでインストールした場合)
        # sys.path.insert(0,'/usr/local/bin/chromedriver') # Homebrewのデフォルトパス
        # または webdriver_manager を使用して自動でダウンロード・管理
        # self.driver_service = ChromeService(ChromeDriverManager().install())
        # 上記はColab向けなので、ローカル環境では直接パスを指定するか、
        # webdriver_managerが自動でパスを見つけることを期待する
        # または、ユーザーが手動でインストールしたchromedriverのパスを環境変数PATHに追加していることを前提とする

        # --- Gemini initialisation for news classification ---
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not set. News country classification will return 'OTHER'.")
            self.gemini_model = None
        else:
            try:
                genai.configure(api_key=api_key)
                chosen = None
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        if m.name == 'models/gemini-2.5-flash-lite-preview-06-17':
                            chosen = m.name; break
                        if m.name == 'models/gemini-2.5-flash-preview-05-20':
                            chosen = m.name; break
                        if 'flash' in m.name:
                            chosen = m.name
                if chosen is None:
                    chosen = genai.list_models()[0].name
                self.gemini_model = genai.GenerativeModel(chosen)
                print(f"Gemini model for classification initialised: {chosen}")
            except Exception as e:
                print(f"Warning: Unable to initialise Gemini model: {e}")
                self.gemini_model = None

    def get_market_data(self):
        """主要指標の直近値、前日比、変化率を取得する"""
        market_data = {}
        ny_tz = pytz.timezone('America/New_York')
        today_ny = datetime.now(ny_tz)
        
        if today_ny.weekday() == 5: # Saturday
            today_ny = today_ny - timedelta(days=1)
        elif today_ny.weekday() == 6: # Sunday
            today_ny = today_ny - timedelta(days=2)
            
        today = today_ny.date()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5: # Saturday or Sunday
            yesterday -= timedelta(days=1)

        for name, ticker in self.tickers.items():
            print(f"--- Market Data: Fetching {name} ({ticker}) ---")
            try:
                if name == "米国2年金利":
                    data = investpy.get_bond_historical_data(bond='U.S. 2Y', from_date=yesterday.strftime('%d/%m/%Y'), to_date=today.strftime('%d/%m/%Y'))
                    if not data.empty and len(data) > 1:
                        current_value = data['Close'].iloc[-1]
                        previous_value = data['Close'].iloc[-2]
                        change = current_value - previous_value
                        change_bp = change * 100
                        change_percent = (change / previous_value) * 100 if previous_value != 0 else 0
                        market_data[name] = {
                            "current": f"{current_value:.3f}",
                            "change": f"{change:.3f}",
                            "change_bp": f"{change_bp:.2f}",
                            "change_percent": f"{change_percent:.2f}%"
                        }
                        print(f"  ✅ {name} data fetched: Current={current_value:.3f}")
                    else:
                        market_data[name] = {"current": "N/A", "change": "N/A", "change_bp": "N/A", "change_percent": "N/A"}
                        print(f"  ❌ {name} data empty or insufficient.")
                    continue
                
                data = yf.download(ticker, start=yesterday - timedelta(days=5), end=today + timedelta(days=1), progress=False) # auto_adjust=Trueがデフォルト
                
                print(f"  Raw data columns for {ticker}: {data.columns}")
                # auto_adjust=Trueの場合、MultiIndexは通常返されないが、念のためチェックは残す
                if isinstance(data.columns, pd.MultiIndex):
                    print(f"  MultiIndex detected for {ticker}. Flattening columns.")
                    data.columns = data.columns.get_level_values(0)
                    print(f"  Flattened columns for {ticker}: {data.columns}")

                if not data.empty and len(data) >= 2:
                    recent_data = data.tail(2)
                    current_value = recent_data['Close'].iloc[-1]
                    previous_value = recent_data['Close'].iloc[-2]
                    change = current_value - previous_value
                    change_percent = (change / previous_value) * 100 if previous_value != 0 else 0
                    
                    market_data[name] = {
                        "current": f"{current_value:.2f}",
                        "change": f"{change:.2f}",
                        "change_percent": f"{change_percent:.2f}%"
                    }
                    print(f"  ✅ {name} data fetched: Current={current_value:.2f}")
                else:
                    market_data[name] = {"current": "N/A", "change": "N/A", "change_percent": "N/A"}
                    print(f"  ❌ {name} data empty or insufficient.")
            except Exception as e:
                print(f"  ❌ Error fetching data for {name} ({ticker}): {e}")
                market_data[name] = {"current": "N/A", "change": "N/A", "change_percent": "N/A"}
        return market_data

    def get_economic_indicators(self):
        """経済指標（過去24時間に発表されたものと、今後24時間に公表予定のもの）を取得する"""
        # economic_data = {"yesterday": [], "today_scheduled": []} # 既存の定義を削除
        
        TARGET_CALENDAR_COUNTRIES = ['united states'] # 定義を追加

        def fetch_and_process_calendar_final():
            # 1. 実行時刻を基準に±24h の期間を計算
            jst = pytz.timezone('Asia/Tokyo')
            now_jst = datetime.now(jst)
            base_time_jst = now_jst  # 実行時刻を基準

            past_limit_jst = base_time_jst - timedelta(hours=24)
            future_limit_jst = base_time_jst + timedelta(hours=24)

            # 休日(週末)の場合は次の営業日まで future_limit を延長
            while future_limit_jst.weekday() >= 5:  # 5=Sat, 6=Sun
                future_limit_jst += timedelta(days=1)

            from_date = past_limit_jst.strftime('%d/%m/%Y')
            to_date = future_limit_jst.strftime('%d/%m/%Y')

            try:
                df_raw = investpy.economic_calendar(
                    from_date=from_date, to_date=to_date, countries=TARGET_CALENDAR_COUNTRIES
                )
            except Exception as e:
                print(f"  ❌ Error: investpyからのデータ取得に失敗しました: {e}")
                return pd.DataFrame()

            if df_raw.empty:
                print("  Warning: 対象期間の経済指標データが見つかりませんでした。")
                return pd.DataFrame()

            # 2. investpy の公表時刻はすべて UTC 基準と仮定し、UTC→JST へ変換する
            df_processed = df_raw.copy()

            # 'time' が 'All Day' や空欄でない行のみを対象にする
            df_processed = df_processed[df_processed['time'].str.contains(':', na=False)].copy()

            # --- investpy の time は既に東京時間とみなす ---
            # 日付と時刻の文字列を結合し、UTCのdatetimeオブジェクトに変換
            df_processed['datetime_utc'] = pd.to_datetime(
                df_processed['date'] + ' ' + df_processed['time'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            ).dt.tz_localize('Asia/Tokyo')

            df_processed.dropna(subset=['datetime_utc'], inplace=True) # 変換失敗行を削除

            # 3. 基準時刻を中心に 24h 前後 (+休日補正) でフィルタリング
            df_filtered = df_processed[
                (df_processed['datetime_utc'] >= past_limit_jst) &
                (df_processed['datetime_utc'] <= future_limit_jst)
            ].copy()

            if df_filtered.empty:
                print("  Warning: 過去24時間～未来24時間の範囲に該当するイベントがありません。")
                return pd.DataFrame()

            # 4. 「発表済み」「発表予定」のステータスを追加
            df_filtered['状態'] = np.where(df_filtered['datetime_utc'] < base_time_jst, '発表済み', '発表予定')

            # 5. 表示用にJSTの日時列を作成
            jst = pytz.timezone('Asia/Tokyo')
            df_filtered['日時(JST)'] = df_filtered['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M')

            # 6. 最終的なカラムを選択・整形
            column_rename_map = {'zone': '国', 'event': 'イベント', 'importance': '重要度', 'actual': '発表値', 'forecast': '予想値', 'previous': '前回値'}
            df_filtered.rename(columns=column_rename_map, inplace=True)
            final_cols = ['状態', '日時(JST)', '国', '重要度', 'イベント', '発表値', '予想値', '前回値']

            df_final = df_filtered[[col for col in final_cols if col in df_filtered.columns]]
            # --- 指標名翻訳と未訳ログ ---
            df_final['イベント_EN'] = df_final['イベント']
            df_final['イベント'] = df_final['イベント_EN'].apply(lambda x: self.indicator_translations.get(x, x))
            untranslated = set(df_final[df_final['イベント'] == df_final['イベント_EN']]['イベント_EN'].unique())
            self._log_untranslated_indicators(untranslated)

            return df_final.sort_values(by='日時(JST)')

        # --- 関数の実行 ---
        df_economic_calendar = fetch_and_process_calendar_final()

        economic_data = {"yesterday": [], "today_scheduled": []} # ここで初期化

        if not df_economic_calendar.empty:
            # '発表済み' のデータを 'yesterday' に
            for _, row in df_economic_calendar[df_economic_calendar['状態'] == '発表済み'].iterrows():
                economic_data["yesterday"].append({
                    "name": row['イベント'],
                    "time": row['日時(JST)'],
                    "previous": row.get('前回値', 'N/A'),
                    "actual": row.get('発表値', 'N/A'),
                    "forecast": row.get('予想値', 'N/A')
                })
            # '発表予定' のデータを 'today_scheduled' に
            for _, row in df_economic_calendar[df_economic_calendar['状態'] == '発表予定'].iterrows():
                economic_data["today_scheduled"].append({
                    "name": row['イベント'],
                    "time": row['日時(JST)'],
                    "previous": row.get('前回値', 'N/A'),
                    "forecast": row.get('予想値', 'N/A')
                })
            print(f"  ✅ Economic indicators fetched: {len(economic_data['yesterday'])} announced, {len(economic_data['today_scheduled'])} scheduled.")
        else:
            print("  ❌ Economic calendar data could not be generated.")
            
        # 未訳インジケーターもここで記録（念のため）
        self._log_untranslated_indicators({row['name'] for cat in economic_data.values() for row in cat if row['name'] not in self.indicator_translations.values()})
        return economic_data

    def _log_untranslated_indicators(self, indicators: set):
        """未訳の経済指標名を重複なくファイルに追記"""
        if not indicators:
            return
        try:
            existing = set()
            if os.path.exists(self.untranslated_file):
                with open(self.untranslated_file, 'r', encoding='utf-8') as f:
                    existing = {line.strip() for line in f if line.strip()}
            new_items = indicators - existing
            if not new_items:
                return
            with open(self.untranslated_file, 'a', encoding='utf-8') as f:
                for item in sorted(new_items):
                    f.write(item + "\n")
            print(f"  🔖 Logged {len(new_items)} untranslated indicator(s).")
        except Exception as e:
            print(f"Warning: Unable to log untranslated indicators: {e}")

    def get_sector_etf_performance(self):
        """米国のセクターETFの変化率を取得する"""
        sector_performance = {}
        ny_tz = pytz.timezone('America/New_York')
        today_ny = datetime.now(ny_tz)
        
        if today_ny.weekday() == 5: # Saturday
            today_ny = today_ny - timedelta(days=1)
        elif today_ny.weekday() == 6: # Sunday
            today_ny = today_ny - timedelta(days=2)
            
        today = today_ny.date()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5: # Saturday or Sunday
            yesterday -= timedelta(days=1)

        for ticker, name in self.sector_etfs.items():
            print(f"--- Sector ETF: Fetching {name} ({ticker}) ---")
            try:
                data = yf.download(ticker, start=yesterday - timedelta(days=5), end=today + timedelta(days=1), progress=False, auto_adjust=False)
                
                print(f"  Raw data columns for {ticker}: {data.columns}")
                if isinstance(data.columns, pd.MultiIndex):
                    print(f"  MultiIndex detected for {ticker}. Flattening columns.")
                    data.columns = data.columns.get_level_values(0)
                    print(f"  Flattened columns for {ticker}: {data.columns}")

                if not data.empty and len(data) >= 2:
                    recent_data = data.tail(2)
                    current_value = recent_data['Close'].iloc[-1]
                    previous_value = recent_data['Close'].iloc[-2]
                    change_percent = ((current_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
                    sector_performance[name] = round(change_percent, 2) if change_percent is not None else None
                    print(f"  ✅ {name} data fetched: {change_percent:.2f}%")
                else:
                    sector_performance[name] = "N/A"
                    print(f"  ❌ {name} data empty or insufficient.")
            except Exception as e:
                print(f"  ❌ Error fetching data for sector ETF {name} ({ticker}): {e}")
                sector_performance[name] = "N/A"
        return sector_performance

    def scrape_reuters_news(self, query: str, hours_limit: int = 24,
                            max_pages: int = 5, items_per_page: int = 20,
                            target_categories: list = None, exclude_keywords: list = None) -> list:
        """ロイターのサイト内検索を利用して記事情報を収集する (Selenium使用)"""
        articles_data, processed_urls = [], set()
        base_search_url = "https://jp.reuters.com/site-search/"
        if target_categories is None: target_categories = []
        if exclude_keywords is None: exclude_keywords = []
        
        driver = None
        print("\n--- ロイター記事のスクレイピング開始 ---")
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.implicitly_wait(15)
            driver.set_page_load_timeout(120)
            jst = pytz.timezone('Asia/Tokyo')
            time_threshold_jst = datetime.now(jst) - timedelta(hours=hours_limit)
            print(f"  [DEBUG] フィルター基準時刻: {time_threshold_jst.strftime('%Y-%m-%d %H:%M')}")

            for page_num in range(max_pages):
                offset = page_num * items_per_page
                search_url = f"{base_search_url}?query={requests.utils.quote(query)}&offset={offset}"
                print(f"  ロイター: ページ {page_num + 1}/{max_pages} を処理中... (URL: {search_url})")
                driver.get(search_url)
                time.sleep(7)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                articles_on_page = soup.find_all('li', attrs={"data-testid": "StoryCard"})

                print(f"  [DEBUG] ページ {page_num + 1} で {len(articles_on_page)} 件の記事候補（liタグ）を発見しました。")
                if not articles_on_page:
                    if page_num == 0: print("    [!] 最初のページで記事候補が全く見つかりません。サイトのHTML構造が変更された可能性が高いです。")
                    break

                for i, article_li in enumerate(articles_on_page):
                    # num_articlesによる制限を削除
                    # if len(articles_data) >= num_articles:
                    #     break
                    # print(f"\n  --- 候補 {i+1} の詳細チェック ---") # DEBUG
                    title, article_url, article_time_jst, category_text = "取得失敗", "取得失敗", None, "不明"

                    title_container = article_li.find('div', class_=re.compile(r'title__title'))
                    link_element = title_container.find('a', attrs={"data-testid": "TitleLink"}) if title_container else None
                    if link_element and link_element.has_attr('href'):
                        article_url = link_element.get('href', '')
                        if article_url.startswith('/'): article_url = "https://jp.reuters.com" + article_url
                    # print(f"    [DEBUG] URL: {article_url}") # DEBUG

                    if link_element:
                        title = link_element.get_text(strip=True)
                    # print(f"    [DEBUG] Title: {title}") # DEBUG

                    time_element = article_li.find('time', attrs={"data-testid": "DateLineText"})
                    if time_element and time_element.has_attr('datetime'):
                        try:
                            dt_utc = datetime.fromisoformat(time_element.get('datetime').replace('Z', '+00:00'))
                            article_time_jst = dt_utc.astimezone(jst)
                            # print(f"    [DEBUG] Time: {article_time_jst.strftime('%Y-%m-%d %H:%M')}") # DEBUG
                        except (ValueError, AttributeError):
                            pass
                    # else:
                        # print("    [DEBUG] Time: timeタグまたはdatetime属性が見つかりません") # DEBUG

                    kicker = article_li.find('span', attrs={"data-testid": "KickerLabel"})
                    # ロイターのカテゴリ名が変動するため、より柔軟にチェック
                    category_text_raw = kicker.get_text(strip=True) if kicker else "不明"
                    category_text = category_text_raw.replace(" category", "").replace("Category", "").strip()
                    # print(f"    [DEBUG] Category: {category_text}") # DEBUG

                    # --- フィルター段階のデバッグ ---
                    if not article_url.startswith('http') or article_url in processed_urls:
                        print(f"    [フィルター] 不正なURL ({article_url}) か、既に処理済みのURLのためスキップします。")
                        continue
                    if article_time_jst is None:
                        print(f"    [フィルター] 日時が取得できなかったためスキップします。")
                        continue
                    if article_time_jst < time_threshold_jst:
                        print(f"    [フィルター] 記事が古いためスキップします。 (記事時刻: {article_time_jst.strftime('%Y-%m-%d %H:%M')})")
                        continue
                    if any(keyword.lower() in title.lower() for keyword in exclude_keywords):
                        print(f"    [フィルター] 除外キーワード ({exclude_keywords}) がタイトルに含まれているためスキップします。 (タイトル: {title})")
                        continue
                    # ユーザーの指定したtarget_categoriesの形式を尊重しつつ、柔軟にチェック
                    if target_categories and not any(tc.lower().replace("category", "").strip() in category_text.lower() for tc in target_categories):
                        print(f"    [フィルター] カテゴリ '{category_text}' がターゲットカテゴリ ({target_categories}) に含まれていないためスキップします。")
                        continue

                    # --- 成功 ---
                    print("    >>> [成功] 全てのチェックを通過しました。記事データをリストに追加します。")
                    body_text = self._scrape_reuters_article_body(article_url) or ""
                    country_code = self.classify_country(f"{title}\n{body_text}")
                    articles_data.append({
                        'title': title, 'url': article_url, 'published_jst': article_time_jst,
                        'category': category_text, 'country': country_code,
                        'body': body_text if body_text else "[本文取得失敗/空]"
                    })
                    processed_urls.add(article_url)

                if len(articles_on_page) < items_per_page: break
                time.sleep(1)
        except Exception as e:
            print(f"  ロイタースクレイピング処理全体でエラーが発生しました: {e}")
            # import traceback; traceback.print_exc() # 詳細なトレースバックが必要な場合
        finally:
            if driver: driver.quit()
        print(f"--- ロイター記事取得完了: {len(articles_data)} 件 ---")
        return articles_data

    def classify_country(self, text: str) -> str:
        """Gemini API を用いて記事の関連国を判定し 2〜3 文字のコードを返す。失敗時は 'OTHER'"""
        if not self.gemini_model:
            return "OTHER"
        prompt = (
            "以下のテキストは経済・マーケット関連ニュースのタイトルと本文です。"\
            "主に関係する国を英語 2 文字(US, JP, CN, EU, UK など) で 1 つだけ回答してください。"\
            "もし特定が難しければ OTHER と答えてください。"\
            "回答は国コードのみを 1 行で出力してください。\n\n---\n" + text[:1800] + "\n---\n")
        try:
            resp = self.gemini_model.generate_content(prompt)
            code = resp.text.strip().upper()
            import re
            m = re.match(r"[A-Z]{2,3}", code)
            return m.group(0) if m else "OTHER"
        except Exception as e:
            print(f"Gemini classify_country error: {e}")
            return "OTHER"

    def _scrape_reuters_article_body(self, article_url: str) -> str:
        """指定されたロイター記事URLから本文を抽出する (requests使用)"""
        print(f"    [DEBUG] Attempting to scrape article body from: {article_url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
            response = requests.get(article_url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.content, 'html.parser')
            
            body_container = soup.find('div', class_='article-body__content__17Yit')
            if not body_container:
                print(f"    [DEBUG] Body container not found for {article_url}. Trying alternative selectors.")
                body_container = soup.find('div', class_='article-body')
                if not body_container:
                    body_container = soup.find('div', class_='text__text__1FZnP')
            
            if not body_container:
                print(f"    [DEBUG] No suitable body container found for {article_url}.")
                return ""

            paragraphs = [p_div.get_text(separator=' ', strip=True) for p_div in body_container.find_all('div', attrs={"data-testid": lambda x: x and x.startswith('paragraph-')})]
            if not paragraphs:
                print(f"    [DEBUG] No paragraphs found with data-testid for {article_url}. Trying generic p tags.")
                paragraphs = [p.get_text(separator=' ', strip=True) for p in body_container.find_all('p')]

            article_text = '\n'.join(paragraphs)
            cleaned_text = re.sub(r'\s+', ' ', article_text).strip()
            
            if not cleaned_text:
                print(f"    [DEBUG] Cleaned article text is empty for {article_url}.")
            else:
                print(f"    [DEBUG] Successfully scraped {len(cleaned_text)} characters from {article_url}.")
            
            return cleaned_text
        except requests.exceptions.RequestException as e:
            print(f"    [DEBUG] Request error scraping article body from {article_url}: {e}")
            return ""
        except Exception as e:
            print(f"    [DEBUG] General error scraping article body from {url}: {e}")
            return ""

    def get_historical_data(self, ticker, period="1y", interval="1d"):
        """
        指定されたティッカーの過去データを取得する。
        period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        """
        print(f"--- Historical Data: Fetching {ticker} ({period}, {interval}) ---")
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if data.empty:
                print(f"  Warning: Empty raw historical data for {ticker}.")
                return pd.DataFrame()

            # MultiIndexを平坦化
            if isinstance(data.columns, pd.MultiIndex):
                print(f"  MultiIndex detected for {ticker}. Flattening columns.")
                data.columns = data.columns.get_level_values(0)
                print(f"  Flattened columns for {ticker}: {data.columns}")

            if not all(col in data.columns for col in required_cols):
                print(f"  Warning: Missing required columns in historical data for {ticker}. Columns: {data.columns.tolist()}")
                return pd.DataFrame()

            for col in ['Open', 'High', 'Low', 'Close']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)
            data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            
            if data.empty:
                print(f"  Warning: Historical data for {ticker} became empty after cleaning.")
                return pd.DataFrame()
            print(f"  ✅ Historical data for {ticker} fetched: {len(data)} rows.")
            # print(f"  Historical data head for {ticker}:\n{data.head()}")
            # print(f"  Historical data info for {ticker}:\n{data.info()}")
            return data
        except Exception as e:
            print(f"  ❌ Error fetching historical data for {ticker}: {e}")
            # import traceback
            # traceback.print_exc()
            return pd.DataFrame()

    def get_intraday_data(self, ticker):
        """
        指定されたティッカーのイントラデイデータを取得する。
        横軸は東京時間で表示されるように処理する。
        """
        jst = pytz.timezone('Asia/Tokyo')
        ny_tz = pytz.timezone('America/New_York')
        utc = pytz.utc

        print(f"--- Intraday Data: Fetching {ticker} ({self.INTRADAY_INTERVAL} for {self.INTRADAY_PERIOD_DAYS} days) ---")
        try:
            df_raw = yf.download(
                ticker, period=f"{self.INTRADAY_PERIOD_DAYS}d", interval=self.INTRADAY_INTERVAL,
                progress=False, auto_adjust=False
            )
            if df_raw.empty:
                print(f"  Warning: Empty raw intraday data for {ticker}.")
                return pd.DataFrame()

            df_cleaned = df_raw.copy()
            print(f"  Raw intraday data columns for {ticker}: {df_cleaned.columns}")

            # MultiIndexを平坦化するロジック
            if isinstance(df_cleaned.columns, pd.MultiIndex):
                print(f"  MultiIndex detected for {ticker}. Flattening columns.")
                df_cleaned.columns = df_cleaned.columns.get_level_values(0)
                print(f"  Flattened columns for {ticker}: {df_cleaned.columns}")

            # データクリーニング処理
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_cleaned.columns for col in ohlcv_cols):
                print(f"  Warning: Missing required OHLCV columns in intraday data for {ticker}. Columns: {df_cleaned.columns.tolist()}")
                return pd.DataFrame()

            for col in ohlcv_cols:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            df_cleaned.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            if df_cleaned.empty:
                print(f"  Warning: Intraday data for {ticker} became empty after cleaning.")
                return pd.DataFrame()

            df_processed = df_cleaned.reset_index()
            datetime_col = 'Datetime' if 'Datetime' in df_processed.columns else 'index'
            df_processed[datetime_col] = pd.to_datetime(df_processed[datetime_col])

            if df_processed[datetime_col].dt.tz is None:
                df_processed[datetime_col] = df_processed[datetime_col].dt.tz_localize(utc)
            else:
                df_processed[datetime_col] = df_processed[datetime_col].dt.tz_convert(utc)

            df_final = pd.DataFrame()

            if ticker in self.ASSET_CLASSES['US_STOCK']:
                print(f"  Info: {ticker} is US_STOCK. Processing for latest NY trading day.")
                df_processed['日時_NY'] = df_processed[datetime_col].dt.tz_convert(ny_tz)
                df_processed['取引日_NY'] = df_processed['日時_NY'].dt.normalize()
                latest_trading_day_ny = df_processed['取引日_NY'].max()
                print(f"  Info: Latest trading day (NY time) for {ticker} is {latest_trading_day_ny.strftime('%Y-%m-%d')}.")
                df_final = df_processed[df_processed['取引日_NY'] == latest_trading_day_ny].copy()

            elif ticker in self.ASSET_CLASSES['24H_ASSET']:
                print(f"  Info: {ticker} is 24H_ASSET. Processing for JST 7am start.")
                df_processed['日時_JST'] = df_processed[datetime_col].dt.tz_convert(jst)
                now_jst = datetime.now(jst)
                today_7am_jst = now_jst.replace(hour=7, minute=0, second=0, microsecond=0)
                start_time_jst = today_7am_jst - timedelta(days=1) if now_jst < today_7am_jst else today_7am_jst
                end_time_jst = start_time_jst + timedelta(days=1)
                print(f"  Info: Extraction period (JST) for {ticker}: {start_time_jst.strftime('%Y-%m-%d %H:%M')} to {end_time_jst.strftime('%Y-%m-%d %H:%M')}.")
                df_final = df_processed[(df_processed['日時_JST'] >= start_time_jst) & (df_processed['日時_JST'] < end_time_jst)].copy()
            else:
                print(f"  Info: {ticker} is neither US_STOCK nor 24H_ASSET. Converting to JST directly.")
                df_final = df_processed.copy()
                df_final['日時'] = df_final[datetime_col].dt.tz_convert(jst)


            if df_final.empty:
                print(f"  Warning: {ticker} の対象期間のデータが抽出できませんでした。")
                return pd.DataFrame()

            if '日時' not in df_final.columns:
                 df_final['日時'] = df_final[datetime_col].dt.tz_convert(jst)

            final_cols = ['日時', 'Open', 'High', 'Low', 'Close', 'Volume']
            final_cols_existing = [col for col in final_cols if col in df_final.columns]
            df_final = df_final[final_cols_existing]

            intraday_chart_data = df_final.set_index('日時')
            print(f"  ✅ Intraday data for {ticker} fetched: {len(intraday_chart_data)} rows.")
            # print(f"  Intraday data head for {ticker}:\n{intraday_chart_data.head()}")
            # print(f"  Intraday data info for {ticker}:\n{intraday_chart_data.info()}")
            return intraday_chart_data

        except Exception as e:
            print(f"  ❌ Error fetching intraday data for {ticker}: {e}")
            # import traceback
            # traceback.print_exc()
            return pd.DataFrame()

if __name__ == "__main__":
    fetcher = DataFetcher()
    
    print("--- Market Data ---")
    market_data = fetcher.get_market_data()
    for name, data in market_data.items():
        print(f"{name}: Current={data['current']}, Change={data.get('change', 'N/A')}, Change_BP={data.get('change_bp', 'N/A')}, Change_Percent={data.get('change_percent', 'N/A')}")

    print("\n--- Economic Indicators ---")
    economic_indicators = fetcher.get_economic_indicators()
    print("Yesterday's Announcements:")
    for item in economic_indicators["yesterday"]:
        print(f"  {item['name']}: Previous={item['previous']}, Actual={item['actual']}, Forecast={item['forecast']}")
    print("Today's Scheduled:")
    for item in economic_indicators["today_scheduled"]:
        print(f"  {item['name']}: Previous={item['previous']}, Actual={item['actual']}, Forecast={item['forecast']}")

    print("\n--- Sector ETF Performance ---")
    sector_performance = fetcher.get_sector_etf_performance()
    for name, perf in sector_performance.items():
        print(f"{name}: {perf}")

    print("\n--- Reuters News ---")
    news = fetcher.scrape_reuters_news(query="米国市場", hours_limit=72, max_pages=5, items_per_page=20, target_categories=["ビジネス", "マーケット", "トップニュース", "ワールド", "テクノロジー", "アジア市場", "経済"], exclude_keywords=["スポーツ", "エンタメ", "五輪", "サッカー", "映画", "将棋", "囲碁", "芸能", "ライフ", "アングル："])
    for article in news:
        print(f"Title: {article['title']}\nLink: {article['link']}\n")

    print("\n--- Historical Data (S&P500 1 year) ---")
    sp500_hist = fetcher.get_historical_data("^GSPC", period="1y")
    print(sp500_hist.head())

    print("\n--- Intraday Data (S&P500 1 day) ---")
    sp500_intraday = fetcher.get_intraday_data("^GSPC", days=1)
    print(sp500_intraday.head())
