#!/usr/bin/env python3
"""
チャート更新スクリプト
移動平均設定に基づいてチャートを再生成する
"""

import json
import sys
import os
from pathlib import Path
from chart_generator import ChartGenerator
from config import Config
import tempfile

def load_chart_settings():
    """設定ファイルから移動平均設定を読み込み"""
    settings_file = Path(__file__).parent / "chart_settings.json"
    
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
    
    # デフォルト設定
    return {
        "maType": "SMA",
        "selectedMAs": ["short", "long"],
        "timestamp": None
    }

def save_chart_settings(settings):
    """設定ファイルに移動平均設定を保存"""
    settings_file = Path(__file__).parent / "chart_settings.json"
    
    try:
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print(f"設定を保存しました: {settings_file}")
    except Exception as e:
        print(f"設定ファイル保存エラー: {e}")

def update_config_with_settings(config, settings):
    """設定に基づいてConfigオブジェクトを更新"""
    
    # 移動平均タイプを更新
    config.DEFAULT_MA_TYPE = settings.get("maType", "SMA")
    
    # 表示する移動平均を更新
    config.DEFAULT_MA_DISPLAY = settings.get("selectedMAs", ["short", "long"])
    
    print(f"移動平均設定更新:")
    print(f"  - タイプ: {config.DEFAULT_MA_TYPE}")
    print(f"  - 表示: {config.DEFAULT_MA_DISPLAY}")
    
    return config

def regenerate_charts_with_settings(settings):
    """設定に基づいてチャートを再生成"""
    try:
        # 設定を読み込み
        config = Config()
        config = update_config_with_settings(config, settings)
        
        # ChartGeneratorを初期化
        chart_gen = ChartGenerator(config=config)
        
        print("チャート再生成を開始...")
        
        # 既存のチャートデータを使用してチャートを再生成
        # Note: 実際のデータは再取得しないで、設定のみ変更してチャートを再描画
        
        # まず、利用可能なチャートファイルを確認
        charts_dir = Path("charts")
        if not charts_dir.exists():
            print("エラー: chartsディレクトリが存在しません")
            return False
        
        # Long-termチャートのみ再生成（移動平均があるのはLong-termチャートのみ）
        longterm_charts = list(charts_dir.glob("*_longterm.html"))
        
        if not longterm_charts:
            print("エラー: Long-termチャートファイルが見つかりません")
            return False
        
        print(f"再生成対象のチャート: {len(longterm_charts)}個")
        
        # データファイルが存在するかチェック
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        
        # 簡易的にいくつかの主要指標でチャートを再生成
        tickers = ["^GSPC", "^NDX", "^DJI", "^SOX", "^TNX", "JPY=X", "EURUSD=X", "BTC-USD", "GC=F", "CL=F", "^VIX"]
        ticker_names = ["S&P500", "NASDAQ100", "ダウ30", "SOX", "米国10年金利", "ドル円", "ユーロドル", "ビットコイン", "ゴールド", "原油", "VIX"]
        
        for ticker, name in zip(tickers, ticker_names):
            try:
                print(f"  {name} チャート再生成中...")
                
                # データを取得（キャッシュがあれば使用）
                longterm_data = fetcher.get_historical_data(ticker, "1y", "1d")
                
                if longterm_data is not None and not longterm_data.empty:
                    # チャートを生成（移動平均設定を適用）
                    filename = f"{name}_longterm.html"
                    chart_gen.generate_longterm_chart_interactive(
                        data=longterm_data,
                        ticker_name=name,
                        filename=filename,
                        ma_keys=settings.get("selectedMAs", ["short", "long"]),
                        ma_type=settings.get("maType", "SMA")
                    )
                    print(f"  ✅ {name} 完了")
                else:
                    print(f"  ⚠️ {name} データなし")
                    
            except Exception as e:
                print(f"  ❌ {name} エラー: {e}")
        
        print("チャート再生成完了!")
        return True
        
    except Exception as e:
        print(f"チャート再生成エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    
    # コマンドライン引数またはstdinから設定を読み込み
    if len(sys.argv) > 1:
        # コマンドライン引数から設定を読み込み
        try:
            settings = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            print("エラー: 無効なJSON設定")
            sys.exit(1)
    else:
        # 既存の設定ファイルを読み込み
        settings = load_chart_settings()
    
    print("=== チャート更新スクリプト ===")
    print(f"移動平均タイプ: {settings.get('maType', 'SMA')}")
    print(f"表示する移動平均: {settings.get('selectedMAs', ['short', 'long'])}")
    
    # 設定を保存
    save_chart_settings(settings)
    
    # チャートを再生成
    success = regenerate_charts_with_settings(settings)
    
    if success:
        print("✅ チャート更新が完了しました")
        sys.exit(0)
    else:
        print("❌ チャート更新に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()