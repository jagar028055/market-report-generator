"""
データ取得モジュール

このモジュールは、従来の巨大なDataFetcherクラスを機能別に分割したものです。
各専用クラスは特定の責任を持ち、テストやメンテナンスが容易になっています。
"""

from .base_fetcher import BaseDataFetcher, DataFetcherFactory
from .market_data_fetcher import MarketDataFetcher
from .news_data_fetcher import NewsDataFetcher
from .economic_data_fetcher import EconomicDataFetcher

# 下位互換性のための統合クラス
class DataFetcher:
    """
    下位互換性のための統合データフェッチャー
    
    従来のDataFetcherクラスと同じインターフェースを提供しつつ、
    内部的には分割されたクラスを使用します。
    """
    
    def __init__(self):
        self.market_fetcher = MarketDataFetcher()
        self.news_fetcher = NewsDataFetcher()
        self.economic_fetcher = EconomicDataFetcher()
        
        # 下位互換性のための属性
        self.tickers = self.market_fetcher.tickers
        self.sector_etfs = self.market_fetcher.sector_etfs
        self.asset_classes = self.market_fetcher.asset_classes
        self.indicator_translations = self.economic_fetcher.indicator_translations
        
        # 設定の参照
        self.data_config = self.market_fetcher.data_config
        
        # Geminiモデル（ニュースフェッチャーから）
        self.gemini_model = self.news_fetcher.gemini_model
        
        # Google Docsクライアント
        self.google_docs_client = self.news_fetcher.google_docs_client
    
    def get_market_data(self):
        """主要指標の直近値、前日比、変化率を取得"""
        return self.market_fetcher.get_market_data()
    
    def get_economic_indicators(self):
        """経済指標（過去24時間に発表されたものと、今後24時間に公表予定のもの）を取得"""
        return self.economic_fetcher.get_economic_indicators()
    
    def get_sector_etf_performance(self):
        """米国のセクターETFの変化率を取得"""
        return self.market_fetcher.get_sector_etf_performance()
    
    def scrape_reuters_news(self, query: str, hours_limit: int = 24, **kwargs):
        """ロイターのサイト内検索を利用して記事情報を収集"""
        return self.news_fetcher.scrape_reuters_news(
            query=query, 
            hours_limit=hours_limit, 
            **kwargs
        )
    
    def get_google_docs_news(self, document_id: str, hours_limit: int = 24):
        """Googleドキュメントからニュース記事を取得"""
        return self.news_fetcher.get_google_docs_news(document_id, hours_limit)
    
    def classify_country(self, text: str):
        """Gemini APIを用いて記事の関連国を判定"""
        return self.news_fetcher.classify_country(text)
    
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d"):
        """指定されたティッカーの過去データを取得"""
        return self.market_fetcher.get_historical_data(ticker, period, interval)
    
    def get_intraday_data(self, ticker: str):
        """指定されたティッカーのイントラデイデータを取得"""
        return self.market_fetcher.get_intraday_data(ticker)
    
    def get_error_summary(self):
        """すべてのフェッチャーのエラー概要を取得"""
        return {
            "market": self.market_fetcher.get_error_summary(),
            "news": self.news_fetcher.get_error_summary(),
            "economic": self.economic_fetcher.get_error_summary()
        }
    
    def clear_error_history(self):
        """すべてのフェッチャーのエラー履歴をクリア"""
        self.market_fetcher.clear_error_history()
        self.news_fetcher.clear_error_history()
        self.economic_fetcher.clear_error_history()


# ファクトリーパターンを使用した取得
def create_data_fetcher(fetcher_type: str = "integrated", **kwargs):
    """
    データフェッチャーを作成
    
    Args:
        fetcher_type: フェッチャーの種類 ("integrated", "market", "news", "economic")
        **kwargs: フェッチャー固有の引数
    
    Returns:
        指定されたタイプのデータフェッチャー
    """
    if fetcher_type == "integrated":
        return DataFetcher()
    elif fetcher_type in ["market", "news", "economic"]:
        return DataFetcherFactory.create_fetcher(fetcher_type, **kwargs)
    else:
        raise ValueError(f"Unknown fetcher type: {fetcher_type}")


# 使用例とユーティリティ関数
def get_all_data():
    """すべてのデータを一括取得"""
    fetcher = DataFetcher()
    
    try:
        # 市場データ
        market_data = fetcher.get_market_data()
        
        # 経済指標
        economic_indicators = fetcher.get_economic_indicators()
        
        # セクターETF
        sector_performance = fetcher.get_sector_etf_performance()
        
        # ニュース（Reuters）
        news_articles = fetcher.scrape_reuters_news(
            query="米国市場 OR 金融 OR 経済",
            hours_limit=24,
            max_pages=3
        )
        
        return {
            "market_data": market_data,
            "economic_indicators": economic_indicators,
            "sector_performance": sector_performance,
            "news_articles": news_articles
        }
        
    except Exception as e:
        raise Exception(f"Failed to fetch all data: {e}")


def get_chart_data(tickers: list = None):
    """チャート用データを取得"""
    fetcher = DataFetcher()
    
    if tickers is None:
        tickers = list(fetcher.tickers.keys())
    
    chart_data = {}
    for name in tickers:
        if name not in ["米国2年金利"]:  # 2年金利は除外
            ticker = fetcher.tickers.get(name)
            if ticker:
                try:
                    chart_data[name] = {
                        "intraday": fetcher.get_intraday_data(ticker),
                        "longterm": fetcher.get_historical_data(ticker, period="1y")
                    }
                except Exception as e:
                    print(f"Error fetching chart data for {name}: {e}")
    
    return chart_data


# モジュールの公開API
__all__ = [
    'BaseDataFetcher',
    'DataFetcherFactory',
    'MarketDataFetcher',
    'NewsDataFetcher',
    'EconomicDataFetcher',
    'DataFetcher',
    'create_data_fetcher',
    'get_all_data',
    'get_chart_data'
]


# 古いインポートパターンとの互換性を保つ
def DataFetcher_legacy():
    """
    従来のインポートパターンとの互換性を保つ関数
    
    使用例:
        from src.data_fetchers import DataFetcher_legacy as DataFetcher
        fetcher = DataFetcher()
    """
    import warnings
    warnings.warn(
        "Direct DataFetcher import is deprecated. Use create_data_fetcher() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return DataFetcher()