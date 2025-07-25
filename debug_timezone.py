#!/usr/bin/env python3
"""
経済指標の時刻調査用デバッグスクリプト
"""

import sys
import os
sys.path.append('src')

from datetime import datetime, timedelta
import pytz
import pandas as pd

def debug_timezone_conversion():
    """タイムゾーン変換のデバッグ"""
    
    print("=== 経済指標時刻デバッグ ===")
    
    # 期待される時刻（外部アプリから）
    expected_times = {
        "Fed Chair Powell Speaks": "21:30",
        "Richmond Manufacturing Index": "23:00"
    }
    
    # 現在の表示時刻（スクリーンショットから）
    current_times = {
        "FRB議長パウエル発言": "13:30",
        "リッチモンド連銀製造業指数": "15:00"
    }
    
    print("期待される時刻（JST）:")
    for event, time in expected_times.items():
        print(f"  {event}: {time}")
    
    print("\n現在の表示時刻（問題のある時刻）:")
    for event, time in current_times.items():
        print(f"  {event}: {time}")
    
    # 時差計算
    print("\n=== 時差分析 ===")
    fed_expected = datetime.strptime("21:30", "%H:%M")
    fed_current = datetime.strptime("13:30", "%H:%M")
    time_diff = fed_expected.hour - fed_current.hour
    
    print(f"期待時刻と現在時刻の差: {time_diff}時間")
    
    if time_diff == 8:
        print("→ 推定原因: UTC時間をJST時間として表示している（JST = UTC + 9時間なので-1時間のずれ）")
        print("→ 対策: investpy APIの時刻をUTCとして扱い、JSTに変換する")
    elif time_diff == 13:
        print("→ 推定原因: 東部時間（EST）をJST時間として表示している")
        print("→ 対策: investpy APIの時刻をETとして扱い、JSTに変換する")
    elif time_diff == 14:
        print("→ 推定原因: 東部時間（EDT、夏時間）をJST時間として表示している")
        print("→ 対策: investpy APIの時刻をETとして扱い、JSTに変換する")
    
    # 修正案の検証
    print("\n=== 修正案の検証 ===")
    
    # 案1: UTC → JST 変換
    test_date = "22/07/2025"
    test_time = "13:30"  # 現在表示されている時刻
    
    # UTC として解釈
    dt_utc = pd.to_datetime(f"{test_date} {test_time}", format='%d/%m/%Y %H:%M').tz_localize('UTC')
    dt_jst_from_utc = dt_utc.tz_convert('Asia/Tokyo')
    
    print(f"案1（UTC→JST）: {test_time} UTC → {dt_jst_from_utc.strftime('%H:%M')} JST")
    
    # 案2: ET → JST 変換  
    dt_et = pd.to_datetime(f"{test_date} {test_time}", format='%d/%m/%Y %H:%M').tz_localize('America/New_York')
    dt_jst_from_et = dt_et.tz_convert('Asia/Tokyo')
    
    print(f"案2（ET→JST）:  {test_time} ET → {dt_jst_from_et.strftime('%H:%M')} JST")
    
    return {
        "utc_to_jst": dt_jst_from_utc.strftime('%H:%M'),
        "et_to_jst": dt_jst_from_et.strftime('%H:%M'),
        "expected": "21:30"
    }

if __name__ == "__main__":
    result = debug_timezone_conversion()
    
    print(f"\n=== 結論 ===")
    if result["utc_to_jst"] == result["expected"]:
        print("✅ UTC → JST 変換が正解")
    elif result["et_to_jst"] == result["expected"]:
        print("✅ ET → JST 変換が正解") 
    else:
        print("❌ 追加調査が必要")