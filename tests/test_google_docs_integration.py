#!/usr/bin/env python3
"""
Google Docs統合のテストスクリプト
実際のGoogle Docsアクセスが困難な場合のためのモック版
"""

import sys
from datetime import datetime
import pytz
from data_fetcher import DataFetcher

# サンプルのGoogle Docsコンテンツ（実際のドキュメント構造に基づく）
SAMPLE_GOOGLE_DOCS_CONTENT = """最終更新: 2025-07-14 09:30

（09:15）米国市場前、フューチャーズは小幅高で推移
https://example.com/news/futures-higher
米国株価指数先物が小幅高で推移している。S&P500先物は0.2%高、ナスダック先物は0.3%高となっている。

（08:45）FRB議事録、次回会合での利下げ示唆
https://example.com/news/fed-minutes
連邦準備制度理事会（FRB）の最新の議事録で、次回会合での利下げの可能性が示唆された。委員の多くが経済指標の軟化を懸念している。

（08:30）米新規失業保険申請件数、予想を下回る
https://example.com/news/jobless-claims
米労働省発表の新規失業保険申請件数は22万件となり、市場予想の23万件を下回った。雇用市場の堅調さが続いている。
"""

class MockGoogleDocsClient:
    """テスト用のGoogle Docsクライアント"""
    
    def fetch_news_articles(self, document_id: str, hours_limit: int = 24) -> list:
        """サンプルデータを返すモック版"""
        
        # サンプルデータをパース
        articles = []
        jst = pytz.timezone('Asia/Tokyo')
        current_time = datetime.now(jst)
        
        # 手動でサンプル記事を作成
        sample_articles = [
            {
                'title': '米国市場前、フューチャーズは小幅高で推移',
                'url': 'https://example.com/news/futures-higher',
                'published_jst': current_time.replace(hour=9, minute=15),
                'category': 'Google Docs',
                'country': 'US',
                'body': '米国株価指数先物が小幅高で推移している。S&P500先物は0.2%高、ナスダック先物は0.3%高となっている。'
            },
            {
                'title': 'FRB議事録、次回会合での利下げ示唆',
                'url': 'https://example.com/news/fed-minutes',
                'published_jst': current_time.replace(hour=8, minute=45),
                'category': 'Google Docs',
                'country': 'US',
                'body': '連邦準備制度理事会（FRB）の最新の議事録で、次回会合での利下げの可能性が示唆された。委員の多くが経済指標の軟化を懸念している。'
            },
            {
                'title': '米新規失業保険申請件数、予想を下回る',
                'url': 'https://example.com/news/jobless-claims',
                'published_jst': current_time.replace(hour=8, minute=30),
                'category': 'Google Docs',
                'country': 'US',
                'body': '米労働省発表の新規失業保険申請件数は22万件となり、市場予想の23万件を下回った。雇用市場の堅調さが続いている。'
            }
        ]
        
        print(f"  ✅ Google Docs（モック版）から {len(sample_articles)} 件の記事を取得しました")
        return sample_articles

def test_integration():
    """Google Docs統合のテスト"""
    print("=== Google Docs統合テスト ===")
    print("注意: 実際のGoogle Docsアクセスのためには認証情報が必要です")
    print("この例では、モック版を使用してデータ形式を確認します")
    
    # モック版のクライアントを作成
    mock_client = MockGoogleDocsClient()
    
    # DataFetcherに手動でモッククライアントを設定
    fetcher = DataFetcher()
    fetcher.google_docs_client = mock_client
    
    print("\n--- Google Docs News Test ---")
    try:
        gdocs_news = fetcher.get_google_docs_news(document_id="test-mock-document")
        
        print(f"\n取得した記事数: {len(gdocs_news)}")
        
        for i, article in enumerate(gdocs_news, 1):
            print(f"\n--- 記事 {i} ---")
            print(f"タイトル: {article['title']}")
            print(f"URL: {article['url']}")
            print(f"時刻: {article['published_jst'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"カテゴリ: {article['category']}")
            print(f"国: {article['country']}")
            print(f"本文: {article['body'][:100]}...")
            
        print("\n=== データ形式確認完了 ===")
        print("実際の統合には以下が必要です:")
        print("1. Google Cloud Platformでプロジェクトを作成")
        print("2. Google Docs APIを有効化")
        print("3. サービスアカウントまたはOAuth認証情報を設定")
        print("4. credentials.jsonファイルをプロジェクトルートに配置")
        print("5. ドキュメントを公開設定にするか、アクセス権限を設定")
        
        return True
        
    except Exception as e:
        print(f"テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)