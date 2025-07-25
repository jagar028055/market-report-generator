#!/usr/bin/env python3
"""
GitHub Actions環境でのタイムゾーン診断スクリプト
"""

import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pytz
import pandas as pd
import investpy
import platform

def diagnose_github_environment():
    """GitHub Actions環境の診断"""
    
    print("=== GitHub Actions 環境診断 ===")
    
    # システム情報
    print(f"Platform: {platform.system()}")
    print(f"Platform version: {platform.version()}")
    print(f"Python version: {platform.python_version()}")
    
    # 環境変数
    print(f"\n=== 環境変数 ===")
    print(f"TZ: {os.environ.get('TZ', 'Not set')}")
    print(f"LANG: {os.environ.get('LANG', 'Not set')}")
    print(f"LC_TIME: {os.environ.get('LC_TIME', 'Not set')}")
    
    # タイムゾーン情報
    print(f"\n=== タイムゾーン情報 ===")
    jst = pytz.timezone('Asia/Tokyo')
    utc = pytz.utc
    et = pytz.timezone('US/Eastern')
    
    now_utc = datetime.now(utc)
    now_jst = datetime.now(jst)
    now_et = datetime.now(et)
    now_local = datetime.now()
    
    print(f"UTC現在時刻: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"JST現在時刻: {now_jst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"ET現在時刻:  {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"ローカル時刻: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 時差確認
    print(f"\n=== 時差確認 ===")
    print(f"JST - UTC = {(now_jst.hour - now_utc.hour) % 24}時間")
    print(f"JST - ET = {(now_jst.hour - now_et.hour) % 24}時間")
    
    return {
        'utc_time': now_utc,
        'jst_time': now_jst,
        'et_time': now_et,
        'local_time': now_local
    }

def test_economic_data_processing():
    """経済データ処理のテスト"""
    
    print(f"\n=== 経済データ処理テスト ===")
    
    try:
        # 経済カレンダーデータを取得（サンプル）
        from src.data_fetchers.economic_data_fetcher import EconomicDataFetcher
        
        fetcher = EconomicDataFetcher()
        
        # 時間範囲の設定をテスト
        jst = pytz.timezone('Asia/Tokyo')
        base_time_jst = datetime.now(jst)
        past_limit_jst = base_time_jst - timedelta(hours=24)
        future_limit_jst = base_time_jst + timedelta(hours=24)
        
        print(f"Base time JST: {base_time_jst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Past limit JST: {past_limit_jst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Future limit JST: {future_limit_jst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # タイムゾーン無しでの比較（問題のある処理）
        print(f"\n=== タイムゾーン処理比較 ===")
        print(f"Base time (tzinfo=None): {base_time_jst.replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Past limit (tzinfo=None): {past_limit_jst.replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 実際に経済指標を取得してテスト
        try:
            economic_data = fetcher.get_economic_indicators(hours_limit=24)
            
            print(f"\n=== 取得された経済指標 ===")
            print(f"発表済み: {len(economic_data.get('yesterday', []))}件")
            print(f"発表予定: {len(economic_data.get('today_scheduled', []))}件")
            
            # 最初の数件の時刻を表示
            if economic_data.get('yesterday'):
                print(f"\n発表済み指標の時刻例:")
                for i, item in enumerate(economic_data['yesterday'][:3]):
                    print(f"  {item['name']}: {item['time']}")
            
            if economic_data.get('today_scheduled'):
                print(f"\n発表予定指標の時刻例:")
                for i, item in enumerate(economic_data['today_scheduled'][:3]):
                    print(f"  {item['name']}: {item['time']}")
                    
        except Exception as e:
            print(f"経済指標取得エラー: {e}")
    
    except Exception as e:
        print(f"処理エラー: {e}")

def test_investpy_raw_data():
    """investpyの生データを確認"""
    
    print(f"\n=== investpy生データテスト ===")
    
    try:
        # 経済カレンダーの生データを取得
        today = datetime.now()
        from_date = (today - timedelta(days=1)).strftime('%d/%m/%Y')
        to_date = (today + timedelta(days=1)).strftime('%d/%m/%Y')
        
        print(f"取得期間: {from_date} - {to_date}")
        
        raw_data = investpy.economic_calendar(
            countries=['united states'],
            from_date=from_date,
            to_date=to_date
        )
        
        if not raw_data.empty:
            print(f"取得件数: {len(raw_data)}件")
            print(f"\n生データサンプル:")
            print(raw_data[['date', 'time', 'event']].head(3).to_string())
            
            # 時刻フォーマットの確認
            time_samples = raw_data['time'].dropna().head(5).tolist()
            print(f"\n時刻フォーマットサンプル: {time_samples}")
        else:
            print("データが取得できませんでした")
            
    except Exception as e:
        print(f"investpy取得エラー: {e}")

if __name__ == "__main__":
    # 環境診断
    env_info = diagnose_github_environment()
    
    # 経済データ処理テスト
    test_economic_data_processing()
    
    # investpy生データテスト
    test_investpy_raw_data()
    
    print(f"\n=== 診断完了 ===")