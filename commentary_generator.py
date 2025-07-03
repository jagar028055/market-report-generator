import google.generativeai as genai
import os
from dotenv import load_dotenv

class CommentaryGenerator:
    def __init__(self):
        load_dotenv() # .envファイルを読み込む
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set or .env file not found.")
        genai.configure(api_key=self.api_key)
        
        # 利用可能なモデルをチェックし、適切なモデルを選択
        self.model = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"Found supported model: {m.name}")
                    if m.name == 'models/gemini-2.5-flash-lite-preview-06-17':
                        self.model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17')
                        break
                    elif m.name == 'models/gemini-2.5-flash-preview-05-20':
                        self.model = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
                        break
                    elif 'flash' in m.name:
                        self.model = genai.GenerativeModel(m.name)
                        break
            if self.model is None:
                raise ValueError("No suitable Gemini model ('gemini-pro' or 'gemini-1.0-pro') found with 'generateContent' capability. Please check available models and your API key/region.")
        except Exception as e:
            raise ValueError(f"Error listing Gemini models: {e}. Please check your API key and network connectivity.")

    def generate_market_commentary(self, news_articles: list, economic_indicators: dict):
        """
        ニュース記事と経済指標データに基づいてマーケットコメントを生成する。
        株式、金利、為替でパラグラフに分ける。
        """
        prompt_parts = [
            "以下の情報に基づいて、前日の米国マーケットコメントを生成してください。\n",
            "コメントは「株式市場」、「金利市場」、「為替市場」の3つのパラグラフに分けてください。\n",
            "経済指標データも考慮し、前日に発表された経済指標についてのコメントも生成してください。\n\n",
            "--- ニュース記事 ---\n"
        ]

        if news_articles:
            for i, article in enumerate(news_articles):
                prompt_parts.append(f"記事 {i+1}: {article['title']} ({article['url']})\n") # 'link'を'url'に変更
        else:
            prompt_parts.append("特筆すべきニュース記事はありません。\n")

        prompt_parts.append("\n--- 経済指標 ---\n")
        if economic_indicators["yesterday"]:
            prompt_parts.append("前日発表された経済指標:\n")
            for item in economic_indicators["yesterday"]:
                prompt_parts.append(f"- {item['name']}: 前回値={item['previous']}, 発表値={item['actual']}, 予想値={item['forecast']}\n")
        else:
            prompt_parts.append("前日発表された経済指標はありません。\n")

        if economic_indicators["today_scheduled"]:
            prompt_parts.append("\n本日発表予定の経済指標:\n")
            for item in economic_indicators["today_scheduled"]:
                prompt_parts.append(f"- {item['name']}: 前回値={item['previous']}, 予想値={item['forecast']}\n")
        else:
            prompt_parts.append("本日発表予定の経済指標はありません。\n")

        prompt_parts.append("\n--- コメント生成 ---\n")

        try:
            response = self.model.generate_content("".join(prompt_parts))
            return response.text
        except Exception as e:
            print(f"Error generating commentary with Gemini API: {e}")
            return "マーケットコメントの生成に失敗しました。"

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_news = [
        {"title": "米株式市場、テクノロジー株が上昇", "link": "http://example.com/news1"},
        {"title": "FRB議長、インフレ抑制に言及", "link": "http://example.com/news2"}
    ]
    dummy_economic_indicators = {
        "yesterday": [
            {"name": "消費者物価指数", "previous": "3.0%", "actual": "3.2%", "forecast": "3.1%"},
            {"name": "小売売上高", "previous": "0.5%", "actual": "0.7%", "forecast": "0.6%"}
        ],
        "today_scheduled": [
            {"name": "製造業PMI", "previous": "52.0", "forecast": "52.5"}
        ]
    }

    # 環境変数にGEMINI_API_KEYを設定してから実行してください
    # 例: export GEMINI_API_KEY="YOUR_API_KEY"
    try:
        comment_gen = CommentaryGenerator()
        commentary = comment_gen.generate_market_commentary(dummy_news, dummy_economic_indicators)
        print("\n--- Generated Market Commentary ---")
        print(commentary)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the GEMINI_API_KEY environment variable.")
