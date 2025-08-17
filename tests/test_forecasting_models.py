"""
予測モデル統合システムテスト

統計予測モデル、機械学習予測、アンサンブル手法、
精度評価、シナリオ分析の包括的テストスイート。
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import os

# テスト対象モジュールのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics.forecasting.statistical_models import ARIMAModel, SARIMAModel
from analytics.forecasting.ml_models import RandomForestModel, XGBoostModel
from analytics.forecasting.ensemble_models import EnsembleForecaster
from analytics.forecasting.accuracy_evaluator import AccuracyEvaluator
from analytics.forecasting.scenario_analyzer import ScenarioAnalyzer


class TestARIMAModel:
    """ARIMAモデルテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.arima = ARIMAModel(max_p=2, max_d=1, max_q=2)
        
        # テストデータ（トレンド付き時系列）
        np.random.seed(42)
        t = np.arange(50)
        trend = 0.5 * t
        noise = np.random.normal(0, 2, 50)
        self.test_data = (100 + trend + noise).tolist()
        
    def test_arima_auto_selection(self):
        """ARIMA自動選択テスト"""
        result = self.arima.auto_arima(self.test_data)
        
        assert 'params' in result
        assert 'aic' in result
        assert 'model' in result
        assert len(result['params']) == 3  # (p, d, q)
        assert isinstance(result['aic'], (int, float))
        
    def test_arima_forecast(self):
        """ARIMA予測テスト"""
        self.arima.auto_arima(self.test_data)
        forecast_result = self.arima.forecast(steps=10)
        
        assert 'forecast' in forecast_result
        assert 'upper_bound' in forecast_result
        assert 'lower_bound' in forecast_result
        assert len(forecast_result['forecast']) == 10
        assert len(forecast_result['upper_bound']) == 10
        assert len(forecast_result['lower_bound']) == 10
        
    def test_arima_diagnostics(self):
        """ARIMA診断統計テスト"""
        self.arima.auto_arima(self.test_data)
        diagnostics = self.arima.get_diagnostics()
        
        assert 'residual_mean' in diagnostics
        assert 'residual_std' in diagnostics
        assert 'normality_test' in diagnostics
        assert 'autocorrelation_lag1' in diagnostics
        assert 'aic' in diagnostics
        assert 'params' in diagnostics
        
    def test_arima_with_short_data(self):
        """短いデータでのARIMAテスト"""
        short_data = self.test_data[:15]  # 短いデータ
        result = self.arima.auto_arima(short_data)
        
        assert result is not None
        forecast_result = self.arima.forecast(steps=5)
        assert len(forecast_result['forecast']) == 5


class TestSARIMAModel:
    """SARIMAモデルテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.sarima = SARIMAModel(seasonal_period=12, max_p=1, max_d=1, max_q=1,
                                max_P=1, max_D=1, max_Q=1)
        
        # 季節性テストデータ
        np.random.seed(42)
        t = np.arange(60)
        trend = 0.3 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)  # 12期の季節性
        noise = np.random.normal(0, 2, 60)
        self.seasonal_data = (100 + trend + seasonal + noise).tolist()
        
    def test_sarima_auto_selection(self):
        """SARIMA自動選択テスト"""
        result = self.sarima.auto_sarima(self.seasonal_data)
        
        assert 'params' in result
        assert 'aic' in result
        assert 'seasonal_strength' in result
        assert 'model' in result
        assert len(result['params']) == 6  # (p, d, q, P, D, Q)
        
    def test_sarima_forecast(self):
        """SARIMA予測テスト"""
        self.sarima.auto_sarima(self.seasonal_data)
        forecast_result = self.sarima.forecast(steps=12)
        
        assert 'forecast' in forecast_result
        assert 'seasonal_component' in forecast_result
        assert 'trend_component' in forecast_result
        assert len(forecast_result['forecast']) == 12
        
    def test_seasonal_diagnostics(self):
        """季節性診断テスト"""
        self.sarima.training_data = np.array(self.seasonal_data)
        diagnostics = self.sarima.get_seasonal_diagnostics()
        
        assert 'seasonal_strength' in diagnostics
        assert 'seasonal_stability' in diagnostics
        assert 'seasonal_period' in diagnostics
        assert diagnostics['seasonal_period'] == 12


class TestRandomForestModel:
    """Random Forestモデルテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.rf_model = RandomForestModel(n_estimators=50, optimize_hyperparams=False)
        
        # テストデータ
        np.random.seed(42)
        t = np.arange(100)
        self.test_data = (100 + 0.5 * t + np.random.normal(0, 5, 100)).tolist()
        
    def test_rf_model_fit(self):
        """Random Forest学習テスト"""
        result = self.rf_model.fit(self.test_data)
        
        assert 'training_metrics' in result
        assert 'feature_importance' in result
        assert 'n_features' in result
        assert self.rf_model.is_fitted
        
        # メトリクス確認
        metrics = result['training_metrics']
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        
    def test_rf_model_predict(self):
        """Random Forest予測テスト"""
        self.rf_model.fit(self.test_data)
        result = self.rf_model.predict(self.test_data, steps=10)
        
        assert 'predictions' in result
        assert 'upper_bound' in result
        assert 'lower_bound' in result
        assert 'feature_importance' in result
        assert len(result['predictions']) == 10
        
    def test_feature_analysis(self):
        """特徴量分析テスト"""
        self.rf_model.fit(self.test_data)
        analysis = self.rf_model.get_feature_analysis()
        
        assert 'feature_importance_ranking' in analysis
        assert 'top_5_features' in analysis
        assert 'total_features' in analysis
        assert 'importance_distribution' in analysis
        
    def test_hyperparameter_optimization(self):
        """ハイパーパラメータ最適化テスト"""
        rf_optimized = RandomForestModel(optimize_hyperparams=True)
        result = rf_optimized.fit(self.test_data)
        
        assert 'best_params' in result
        assert rf_optimized.is_fitted


class TestXGBoostModel:
    """XGBoostモデルテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.xgb_model = XGBoostModel(n_estimators=50, optimize_hyperparams=False)
        
        # テストデータ
        np.random.seed(42)
        t = np.arange(80)
        self.test_data = (100 + 0.8 * t + np.random.normal(0, 3, 80)).tolist()
        
    def test_xgb_model_fit(self):
        """XGBoost学習テスト"""
        result = self.xgb_model.fit(self.test_data)
        
        assert 'training_metrics' in result
        assert 'feature_importance' in result
        assert 'n_trees' in result
        assert self.xgb_model.is_fitted
        assert result['n_trees'] > 0
        
    def test_xgb_model_predict(self):
        """XGBoost予測テスト"""
        self.xgb_model.fit(self.test_data)
        result = self.xgb_model.predict(self.test_data, steps=8)
        
        assert 'predictions' in result
        assert 'n_trees_used' in result
        assert len(result['predictions']) == 8
        
    def test_boosting_analysis(self):
        """ブースティング分析テスト"""
        self.xgb_model.fit(self.test_data)
        analysis = self.xgb_model.get_boosting_analysis()
        
        assert 'n_trees' in analysis
        assert 'tree_weight_stats' in analysis
        assert 'model_complexity' in analysis
        assert analysis['n_trees'] > 0


class TestEnsembleForecaster:
    """アンサンブル予測テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.ensemble = EnsembleForecaster(
            ensemble_method='weighted_average',
            optimize_weights=True
        )
        
        # テストデータ（十分な長さ）
        np.random.seed(42)
        t = np.arange(120)
        trend = 0.4 * t
        seasonal = 3 * np.sin(2 * np.pi * t / 24)
        noise = np.random.normal(0, 4, 120)
        self.test_data = (100 + trend + seasonal + noise).tolist()
        
    def test_ensemble_fit(self):
        """アンサンブル学習テスト"""
        result = self.ensemble.fit(self.test_data)
        
        assert 'successful_models' in result
        assert 'model_weights' in result
        assert 'model_performance' in result
        assert 'ensemble_method' in result
        assert self.ensemble.is_fitted
        assert len(result['successful_models']) > 0
        
    def test_ensemble_forecast(self):
        """アンサンブル予測テスト"""
        self.ensemble.fit(self.test_data)
        result = self.ensemble.forecast(steps=15)
        
        assert 'ensemble_forecast' in result
        assert 'individual_predictions' in result
        assert 'confidence_interval' in result
        assert 'prediction_analysis' in result
        assert len(result['ensemble_forecast']) == 15
        
    def test_stacking_ensemble(self):
        """スタッキングアンサンブルテスト"""
        stacking_ensemble = EnsembleForecaster(ensemble_method='stacking')
        stacking_ensemble.fit(self.test_data)
        result = stacking_ensemble.forecast(steps=10)
        
        assert 'ensemble_forecast' in result
        assert len(result['ensemble_forecast']) == 10
        
    def test_ensemble_summary(self):
        """アンサンブルサマリーテスト"""
        self.ensemble.fit(self.test_data)
        summary = self.ensemble.get_ensemble_summary()
        
        assert 'ensemble_method' in summary
        assert 'model_weights' in summary
        assert 'active_models' in summary


class TestAccuracyEvaluator:
    """精度評価システムテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.evaluator = AccuracyEvaluator(test_ratio=0.2, rolling_window=30)
        
        # テストデータとモデル
        np.random.seed(42)
        t = np.arange(100)
        self.test_data = (100 + 0.3 * t + np.random.normal(0, 3, 100)).tolist()
        
        # テスト用モデル（簡易）
        class MockModel:
            def __init__(self):
                self.is_fitted = False
                
            def fit(self, data):
                self.is_fitted = True
                self.last_value = data[-1]
                
            def forecast(self, steps):
                return {'forecast': [self.last_value + i * 0.5 for i in range(steps)]}
                
        self.mock_model = MockModel()
        
    def test_simple_split_evaluation(self):
        """単純分割評価テスト"""
        result = self.evaluator.evaluate_model(
            self.test_data, self.mock_model, ['simple_split']
        )
        
        assert 'simple_split' in result
        split_result = result['simple_split']
        assert 'metrics' in split_result
        assert 'residual_statistics' in split_result
        assert 'statistical_tests' in split_result
        
    def test_rolling_window_evaluation(self):
        """ローリングウィンドウ評価テスト"""
        result = self.evaluator.evaluate_model(
            self.test_data, self.mock_model, ['rolling_window']
        )
        
        assert 'rolling_window' in result
        rolling_result = result['rolling_window']
        assert 'overall_metrics' in rolling_result
        assert 'window_metrics' in rolling_result
        
    def test_time_series_cv(self):
        """時系列交差検証テスト"""
        result = self.evaluator.evaluate_model(
            self.test_data, self.mock_model, ['time_series_cv']
        )
        
        assert 'time_series_cv' in result
        cv_result = result['time_series_cv']
        assert 'cv_summary' in cv_result
        assert 'fold_results' in cv_result
        
    def test_model_comparison(self):
        """モデル比較テスト"""
        models = {
            'model1': self.mock_model,
            'model2': MockModel()
        }
        
        result = self.evaluator.compare_models(self.test_data, models)
        
        assert 'individual_results' in result
        assert 'model_rankings' in result
        assert 'summary' in result


class TestScenarioAnalyzer:
    """シナリオ分析テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.analyzer = ScenarioAnalyzer(n_simulations=100)  # テスト用に少なく
        
        # テストデータとモデル
        np.random.seed(42)
        t = np.arange(80)
        self.test_data = (100 + 0.4 * t + np.random.normal(0, 5, 80)).tolist()
        
        # テスト用モデル
        class MockForecastModel:
            def fit(self, data):
                self.data_mean = np.mean(data)
                
            def forecast(self, steps):
                return {'forecast': [self.data_mean + i * 0.3 for i in range(steps)]}
                
        self.mock_forecast_model = MockForecastModel()
        
    def test_scenario_analysis(self):
        """シナリオ分析テスト"""
        self.mock_forecast_model.fit(self.test_data)
        
        result = self.analyzer.analyze_scenarios(
            self.test_data, 
            self.mock_forecast_model
        )
        
        assert 'analysis_timestamp' in result
        assert 'scenario_config' in result
        assert 'risk_summary' in result
        
    def test_monte_carlo_simulation(self):
        """モンテカルロシミュレーションテスト"""
        self.mock_forecast_model.fit(self.test_data)
        
        config = {
            'monte_carlo': True,
            'stress_test': False,
            'sensitivity_analysis': False,
            'forecast_steps': 5
        }
        
        result = self.analyzer.analyze_scenarios(
            self.test_data, 
            self.mock_forecast_model,
            config
        )
        
        assert 'monte_carlo' in result
        mc_result = result['monte_carlo']
        assert 'simulations' in mc_result
        assert 'statistics' in mc_result
        assert mc_result['n_simulations'] == 100
        
    def test_stress_test(self):
        """ストレステストテスト"""
        self.mock_forecast_model.fit(self.test_data)
        
        config = {
            'monte_carlo': False,
            'stress_test': True,
            'sensitivity_analysis': False
        }
        
        result = self.analyzer.analyze_scenarios(
            self.test_data,
            self.mock_forecast_model,
            config
        )
        
        assert 'stress_test' in result
        
    def test_sensitivity_analysis(self):
        """感度分析テスト"""
        self.mock_forecast_model.fit(self.test_data)
        
        config = {
            'monte_carlo': False,
            'stress_test': False,
            'sensitivity_analysis': True
        }
        
        result = self.analyzer.analyze_scenarios(
            self.test_data,
            self.mock_forecast_model,
            config
        )
        
        assert 'sensitivity_analysis' in result
        
    def test_scenario_report(self):
        """シナリオレポートテスト"""
        self.mock_forecast_model.fit(self.test_data)
        
        # まず分析実行
        self.analyzer.analyze_scenarios(
            self.test_data,
            self.mock_forecast_model
        )
        
        report = self.analyzer.get_scenario_report()
        
        assert 'executive_summary' in report
        assert 'detailed_analysis' in report
        assert 'recommendations' in report
        assert 'report_timestamp' in report


class TestIntegrationForecastingSystem:
    """予測システム統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        # より長いテストデータ
        np.random.seed(42)
        t = np.arange(200)
        trend = 0.2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 50)
        noise = np.random.normal(0, 5, 200)
        self.integration_data = (100 + trend + seasonal + noise).tolist()
        
    def test_full_forecasting_pipeline(self):
        """完全予測パイプラインテスト"""
        # 1. アンサンブルモデル学習
        ensemble = EnsembleForecaster()
        ensemble_result = ensemble.fit(self.integration_data)
        assert ensemble.is_fitted
        
        # 2. 予測実行
        forecast_result = ensemble.forecast(steps=20)
        assert len(forecast_result['ensemble_forecast']) == 20
        
        # 3. 精度評価
        evaluator = AccuracyEvaluator()
        eval_result = evaluator.evaluate_model(
            self.integration_data, ensemble, ['simple_split']
        )
        assert 'simple_split' in eval_result
        
        # 4. シナリオ分析
        analyzer = ScenarioAnalyzer(n_simulations=50)  # テスト用
        scenario_result = analyzer.analyze_scenarios(
            self.integration_data, ensemble
        )
        assert 'risk_summary' in scenario_result
        
    def test_model_comparison_workflow(self):
        """モデル比較ワークフローテスト"""
        # 個別モデル作成
        arima = ARIMAModel(max_p=2, max_d=1, max_q=2)
        rf = RandomForestModel(n_estimators=30, optimize_hyperparams=False)
        
        # 学習
        arima.auto_arima(self.integration_data)
        rf.fit(self.integration_data)
        
        # 比較評価
        evaluator = AccuracyEvaluator()
        models = {
            'ARIMA': arima,
            'RandomForest': rf
        }
        
        comparison_result = evaluator.compare_models(self.integration_data, models)
        
        assert 'individual_results' in comparison_result
        assert 'model_rankings' in comparison_result
        assert len(comparison_result['individual_results']) == 2
        
    def test_performance_metrics(self):
        """パフォーマンスメトリクステスト"""
        import time
        
        # 高速モデルテスト
        start_time = time.time()
        
        rf_fast = RandomForestModel(n_estimators=10, optimize_hyperparams=False)
        rf_fast.fit(self.integration_data[:100])  # 短いデータ
        forecast = rf_fast.predict(self.integration_data[:100], steps=5)
        
        execution_time = time.time() - start_time
        
        assert execution_time < 10.0  # 10秒以内
        assert len(forecast['predictions']) == 5
        
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 不正データテスト
        invalid_data = [float('inf'), float('nan'), 1, 2, 3]
        
        with pytest.raises(ValueError):
            ensemble = EnsembleForecaster()
            ensemble.fit(invalid_data)
            
        # 短すぎるデータテスト
        short_data = [1, 2, 3]
        
        with pytest.raises(ValueError):
            analyzer = ScenarioAnalyzer()
            analyzer.analyze_scenarios(short_data, None)


if __name__ == '__main__':
    # テスト実行用
    import subprocess
    
    print("予測モデル統合システムテスト開始...")
    
    # pytest実行
    result = subprocess.run([
        'python', '-m', 'pytest', __file__, '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("エラー:", result.stderr)
        
    print(f"テスト終了: 終了コード={result.returncode}")