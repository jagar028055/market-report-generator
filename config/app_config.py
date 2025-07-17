"""
アプリケーション設定ファイル
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """アプリケーション設定クラス"""
    
    # プロジェクトルートディレクトリ
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Google Docs設定
    GOOGLE_DOCS_ID = os.getenv("GOOGLE_DOCS_ID")
    NEWS_HOURS_LIMIT = 24
    
    # 除外する銘柄
    EXCLUDED_TICKERS = ["米国2年金利"]
    
    # チャート設定
    CHART_PERIOD = "1y"
    
    # 出力ファイル設定
    OUTPUT_FILENAME = "index.html"
    CHARTS_DIR = "charts"
    SUCCESS_LOG = "success.log"
    ERROR_LOG = "error.log"
    
    # セクター別ETFチャート設定
    SECTOR_CHART_FILENAME = "sector_performance_chart.html"
    
    # 処理ステップ名
    PROCESSING_STEPS = [
        "データ取得中",
        "チャート生成中", 
        "AIコメント生成中",
        "セクター別ETFチャート生成中",
        "HTMLレポート生成中"
    ]
    
    @classmethod
    def get_charts_output_dir(cls, base_dir: str) -> str:
        """チャート出力ディレクトリパスを取得"""
        return os.path.join(base_dir, cls.CHARTS_DIR)
    
    @classmethod
    def get_output_file_path(cls, base_dir: str) -> str:
        """出力ファイルパスを取得"""
        return os.path.join(base_dir, cls.OUTPUT_FILENAME)
