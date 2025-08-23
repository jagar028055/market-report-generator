#!/usr/bin/env python3
"""
Market Report Generator Web Server Test
新しいUI機能の簡易テスト
"""
import os
import sys
from datetime import datetime

# 環境変数設定（テスト用）
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('ALPHA_VANTAGE_API_KEY', 'test-key')

def test_imports():
    """基本インポートテスト"""
    print("=== インポートテスト開始 ===")
    
    try:
        from src.api import create_api_routes
        print("✓ API routes import success")
    except Exception as e:
        print(f"✗ API routes import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask import success")
    except Exception as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        from src.core.data_fetcher import DataFetcher
        print("✓ DataFetcher import success")
    except Exception as e:
        print(f"✗ DataFetcher import failed: {e}")
        return False
    
    print("=== インポートテスト完了 ===\n")
    return True

def test_flask_app():
    """Flaskアプリケーション作成テスト"""
    print("=== Flaskアプリテスト開始 ===")
    
    try:
        from flask import Flask
        from src.api import create_api_routes
        
        app = Flask(__name__, 
                   template_folder='templates',
                   static_folder='static')
        
        # APIルート追加テスト
        create_api_routes(app)
        print("✓ API routes created successfully")
        
        # ルート確認
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(f"{rule.rule} [{', '.join(rule.methods)}]")
        
        print(f"✓ {len(routes)} routes registered:")
        for route in sorted(routes):
            print(f"  - {route}")
        
        print("=== Flaskアプリテスト完了 ===\n")
        return True
        
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

def test_api_fallback():
    """APIフォールバック機能テスト"""
    print("=== APIフォールバックテスト開始 ===")
    
    try:
        from src.api.routes import (
            FallbackForecaster, 
            FallbackAccuracyEvaluator,
            FallbackRiskGenerator,
            FallbackMonteCarloVisualizer
        )
        
        # フォールバック予測テスト
        forecaster = FallbackForecaster()
        test_data = [
            {'date': '2024-01-01', 'close': 100},
            {'date': '2024-01-02', 'close': 102},
            {'date': '2024-01-03', 'close': 98},
            {'date': '2024-01-04', 'close': 105},
            {'date': '2024-01-05', 'close': 103}
        ]
        
        prediction = forecaster.predict_simple('AAPL', 7, test_data)
        print(f"✓ Fallback prediction: {prediction}")
        
        # フォールバック精度評価テスト
        evaluator = FallbackAccuracyEvaluator()
        accuracy = evaluator.evaluate_model('ensemble', 'AAPL', prediction)
        print(f"✓ Fallback accuracy: {accuracy}")
        
        # フォールバックリスクテスト
        risk_gen = FallbackRiskGenerator()
        var_cvar = risk_gen.calculate_var_cvar('AAPL', test_data, 0.95)
        print(f"✓ Fallback VaR/CVaR: {var_cvar}")
        
        # フォールバックモンテカルロテスト
        monte_carlo = FallbackMonteCarloVisualizer()
        sim_result = monte_carlo.run_simulation('AAPL', test_data, 30, 100)
        print(f"✓ Fallback Monte Carlo simulation: {len(sim_result['final_returns'])} results")
        
        print("=== APIフォールバックテスト完了 ===\n")
        return True
        
    except Exception as e:
        print(f"✗ API fallback test failed: {e}")
        return False

def main():
    """メインテスト関数"""
    print(f"Market Report Generator Web Server Test")
    print(f"Test started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}\n")
    
    # 各テストの実行
    tests = [
        ("Import Test", test_imports),
        ("Flask App Test", test_flask_app),
        ("API Fallback Test", test_api_fallback)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Web server is ready to run.")
        print("\nTo start the web server, run:")
        print("  python main.py server")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)