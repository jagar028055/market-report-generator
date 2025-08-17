"""
予測精度評価・バックテストシステム

時系列予測モデルの精度評価、バックテスト、パフォーマンス分析を提供。
ローリングウィンドウ評価、時系列交差検証、多様な評価指標、
統計的有意性検定を含む包括的な評価システム。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """
    予測精度評価システム
    
    バックテスト、ローリングウィンドウ評価、時系列交差検証、
    統計的検定、モデル比較を含む包括的な評価機能。
    """
    
    def __init__(self, test_ratio: float = 0.2, rolling_window: int = 50,
                 significance_level: float = 0.05):
        """
        Args:
            test_ratio: テストデータ比率
            rolling_window: ローリングウィンドウサイズ
            significance_level: 統計的有意水準
        """
        self.test_ratio = test_ratio
        self.rolling_window = rolling_window
        self.significance_level = significance_level
        self.evaluation_results = {}
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """基本評価指標計算"""
        metrics = {}
        
        # 基本指標
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = float('inf')
            
        # sMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        non_zero_denom = denominator != 0
        if np.any(non_zero_denom):
            smape = np.mean(np.abs(y_true[non_zero_denom] - y_pred[non_zero_denom]) / denominator[non_zero_denom]) * 100
            metrics['smape'] = smape
        else:
            metrics['smape'] = 0.0
            
        # 方向精度 (Directional Accuracy)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
            metrics['direction_accuracy'] = direction_accuracy
        else:
            metrics['direction_accuracy'] = 50.0
            
        # Theil's U統計量
        if np.var(y_true) > 0:
            naive_forecast = np.roll(y_true, 1)[1:]  # 前日値予測
            actual_diff = y_true[1:]
            pred_diff = y_pred[1:]
            
            mse_model = mean_squared_error(actual_diff, pred_diff)
            mse_naive = mean_squared_error(actual_diff, naive_forecast)
            
            if mse_naive > 0:
                theil_u = np.sqrt(mse_model) / np.sqrt(mse_naive)
                metrics['theil_u'] = theil_u
            else:
                metrics['theil_u'] = 1.0
        else:
            metrics['theil_u'] = 1.0
            
        return metrics
        
    def _rolling_window_evaluation(self, data: List[float], 
                                 forecaster: Any, steps: int = 1) -> Dict[str, Any]:
        """ローリングウィンドウ評価"""
        data_array = np.array(data)
        n = len(data_array)
        
        if n < self.rolling_window + steps:
            raise ValueError(f"データ数不足: {n} < {self.rolling_window + steps}")
            
        predictions = []
        actuals = []
        evaluation_windows = []
        
        start_idx = self.rolling_window
        end_idx = n - steps + 1
        
        logger.info(f"ローリングウィンドウ評価開始: {start_idx}から{end_idx}まで")
        
        for i in range(start_idx, end_idx):
            try:
                # 学習データ
                train_data = data_array[i - self.rolling_window:i].tolist()
                
                # 実際値
                actual_values = data_array[i:i + steps].tolist()
                
                # モデル学習・予測
                if hasattr(forecaster, 'fit'):
                    forecaster.fit(train_data)
                    
                if hasattr(forecaster, 'forecast'):
                    forecast_result = forecaster.forecast(steps)
                    if isinstance(forecast_result, dict):
                        pred_values = forecast_result.get('forecast', forecast_result.get('ensemble_forecast', []))
                    else:
                        pred_values = forecast_result
                elif hasattr(forecaster, 'predict'):
                    predict_result = forecaster.predict(train_data, steps)
                    if isinstance(predict_result, dict):
                        pred_values = predict_result.get('predictions', [])
                    else:
                        pred_values = predict_result
                else:
                    raise ValueError("予測メソッドが見つかりません")
                    
                # 結果保存
                if len(pred_values) >= len(actual_values):
                    predictions.extend(pred_values[:len(actual_values)])
                    actuals.extend(actual_values)
                    evaluation_windows.append({
                        'window_start': i - self.rolling_window,
                        'window_end': i,
                        'prediction_start': i,
                        'prediction_end': i + steps,
                        'predictions': pred_values[:len(actual_values)],
                        'actuals': actual_values
                    })
                    
            except Exception as e:
                logger.warning(f"ローリングウィンドウ{i}での評価失敗: {e}")
                continue
                
        if not predictions:
            raise ValueError("ローリングウィンドウ評価で予測値が取得できませんでした")
            
        # 全体メトリクス計算
        overall_metrics = self._calculate_metrics(np.array(actuals), np.array(predictions))
        
        # ウィンドウ別メトリクス計算
        window_metrics = []
        for window in evaluation_windows:
            window_metric = self._calculate_metrics(
                np.array(window['actuals']), 
                np.array(window['predictions'])
            )
            window_metric.update(window)
            window_metrics.append(window_metric)
            
        return {
            'overall_metrics': overall_metrics,
            'window_metrics': window_metrics,
            'total_predictions': len(predictions),
            'total_windows': len(evaluation_windows)
        }
        
    def _time_series_cv(self, data: List[float], forecaster: Any, 
                       n_splits: int = 5, steps: int = 1) -> Dict[str, Any]:
        """時系列交差検証"""
        data_array = np.array(data)
        n = len(data_array)
        
        # 分割サイズ計算
        min_train_size = max(20, self.rolling_window)
        test_size = max(steps, n // (n_splits + 2))
        
        if n < min_train_size + test_size:
            raise ValueError(f"データ数不足: CV実行不可")
            
        cv_results = []
        
        for fold in range(n_splits):
            try:
                # 分割点計算
                train_end = min_train_size + fold * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n)
                
                if test_end <= test_start:
                    break
                    
                # データ分割
                train_data = data_array[:train_end].tolist()
                test_data = data_array[test_start:test_end].tolist()
                
                # モデル学習
                if hasattr(forecaster, 'fit'):
                    forecaster.fit(train_data)
                    
                # 予測実行
                predict_steps = len(test_data)
                if hasattr(forecaster, 'forecast'):
                    forecast_result = forecaster.forecast(predict_steps)
                    if isinstance(forecast_result, dict):
                        predictions = forecast_result.get('forecast', forecast_result.get('ensemble_forecast', []))
                    else:
                        predictions = forecast_result
                elif hasattr(forecaster, 'predict'):
                    predict_result = forecaster.predict(train_data, predict_steps)
                    if isinstance(predict_result, dict):
                        predictions = predict_result.get('predictions', [])
                    else:
                        predictions = predict_result
                else:
                    continue
                    
                # 予測数調整
                predictions = predictions[:len(test_data)]
                if len(predictions) < len(test_data):
                    # 不足分は最後の値で埋める
                    last_pred = predictions[-1] if predictions else test_data[0]
                    predictions.extend([last_pred] * (len(test_data) - len(predictions)))
                    
                # メトリクス計算
                fold_metrics = self._calculate_metrics(np.array(test_data), np.array(predictions))
                fold_metrics.update({
                    'fold': fold,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                
                cv_results.append(fold_metrics)
                
            except Exception as e:
                logger.warning(f"CV fold {fold}での評価失敗: {e}")
                continue
                
        if not cv_results:
            raise ValueError("時系列交差検証で有効な結果が得られませんでした")
            
        # 統計サマリー
        metric_names = ['mse', 'rmse', 'mae', 'r2', 'mape', 'smape', 'direction_accuracy', 'theil_u']
        cv_summary = {}
        
        for metric in metric_names:
            values = [fold[metric] for fold in cv_results if metric in fold]
            if values:
                cv_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
                
        return {
            'cv_summary': cv_summary,
            'fold_results': cv_results,
            'n_successful_folds': len(cv_results)
        }
        
    def _statistical_tests(self, residuals: np.ndarray) -> Dict[str, Any]:
        """統計的検定"""
        tests = {}
        
        # 正規性検定 (Shapiro-Wilk)
        try:
            if len(residuals) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                tests['normality'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > self.significance_level
                }
            else:
                tests['normality'] = {'test': 'Shapiro-Wilk', 'error': 'insufficient_data'}
        except Exception as e:
            tests['normality'] = {'test': 'Shapiro-Wilk', 'error': str(e)}
            
        # 自己相関検定 (Ljung-Box簡易版)
        try:
            if len(residuals) > 10:
                # ラグ1自己相関
                lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                
                # ラグ5自己相関
                if len(residuals) > 5:
                    lag5_corr = np.corrcoef(residuals[:-5], residuals[5:])[0, 1]
                else:
                    lag5_corr = 0.0
                    
                tests['autocorrelation'] = {
                    'test': 'Autocorrelation',
                    'lag1_correlation': lag1_corr if not np.isnan(lag1_corr) else 0.0,
                    'lag5_correlation': lag5_corr if not np.isnan(lag5_corr) else 0.0,
                    'significant_autocorr': abs(lag1_corr) > 0.2 if not np.isnan(lag1_corr) else False
                }
            else:
                tests['autocorrelation'] = {'test': 'Autocorrelation', 'error': 'insufficient_data'}
        except Exception as e:
            tests['autocorrelation'] = {'test': 'Autocorrelation', 'error': str(e)}
            
        # 等分散性検定 (簡易版)
        try:
            if len(residuals) > 20:
                mid_point = len(residuals) // 2
                first_half_var = np.var(residuals[:mid_point])
                second_half_var = np.var(residuals[mid_point:])
                
                variance_ratio = max(first_half_var, second_half_var) / (min(first_half_var, second_half_var) + 1e-8)
                
                tests['homoscedasticity'] = {
                    'test': 'Variance Ratio',
                    'variance_ratio': variance_ratio,
                    'first_half_var': first_half_var,
                    'second_half_var': second_half_var,
                    'is_homoscedastic': variance_ratio < 2.0
                }
            else:
                tests['homoscedasticity'] = {'test': 'Variance Ratio', 'error': 'insufficient_data'}
        except Exception as e:
            tests['homoscedasticity'] = {'test': 'Variance Ratio', 'error': str(e)}
            
        return tests
        
    def evaluate_model(self, data: List[float], forecaster: Any, 
                      evaluation_methods: List[str] = None) -> Dict[str, Any]:
        """
        包括的モデル評価
        
        Args:
            data: 時系列データ
            forecaster: 予測モデル（fit, forecast/predictメソッド必須）
            evaluation_methods: 評価手法リスト
            
        Returns:
            評価結果
        """
        if evaluation_methods is None:
            evaluation_methods = ['simple_split', 'rolling_window', 'time_series_cv']
            
        logger.info(f"モデル評価開始: 手法={evaluation_methods}")
        
        data_array = np.array(data)
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'data_size': len(data),
            'evaluation_methods': evaluation_methods
        }
        
        # 単純分割評価
        if 'simple_split' in evaluation_methods:
            try:
                results['simple_split'] = self._simple_split_evaluation(data, forecaster)
            except Exception as e:
                logger.warning(f"単純分割評価失敗: {e}")
                results['simple_split'] = {'error': str(e)}
                
        # ローリングウィンドウ評価
        if 'rolling_window' in evaluation_methods:
            try:
                results['rolling_window'] = self._rolling_window_evaluation(data, forecaster)
            except Exception as e:
                logger.warning(f"ローリングウィンドウ評価失敗: {e}")
                results['rolling_window'] = {'error': str(e)}
                
        # 時系列交差検証
        if 'time_series_cv' in evaluation_methods:
            try:
                results['time_series_cv'] = self._time_series_cv(data, forecaster)
            except Exception as e:
                logger.warning(f"時系列交差検証失敗: {e}")
                results['time_series_cv'] = {'error': str(e)}
                
        return results
        
    def _simple_split_evaluation(self, data: List[float], forecaster: Any) -> Dict[str, Any]:
        """単純分割評価"""
        data_array = np.array(data)
        split_point = int(len(data_array) * (1 - self.test_ratio))
        
        train_data = data_array[:split_point].tolist()
        test_data = data_array[split_point:].tolist()
        
        if len(test_data) == 0:
            raise ValueError("テストデータが空です")
            
        # モデル学習
        if hasattr(forecaster, 'fit'):
            forecaster.fit(train_data)
            
        # 予測実行
        test_steps = len(test_data)
        if hasattr(forecaster, 'forecast'):
            forecast_result = forecaster.forecast(test_steps)
            if isinstance(forecast_result, dict):
                predictions = forecast_result.get('forecast', forecast_result.get('ensemble_forecast', []))
            else:
                predictions = forecast_result
        elif hasattr(forecaster, 'predict'):
            predict_result = forecaster.predict(train_data, test_steps)
            if isinstance(predict_result, dict):
                predictions = predict_result.get('predictions', [])
            else:
                predictions = predict_result
        else:
            raise ValueError("予測メソッドが見つかりません")
            
        # 予測数調整
        predictions = predictions[:len(test_data)]
        if len(predictions) < len(test_data):
            last_pred = predictions[-1] if predictions else test_data[0]
            predictions.extend([last_pred] * (len(test_data) - len(predictions)))
            
        # メトリクス計算
        metrics = self._calculate_metrics(np.array(test_data), np.array(predictions))
        
        # 残差分析
        residuals = np.array(test_data) - np.array(predictions)
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # 統計的検定
        statistical_tests = self._statistical_tests(residuals)
        
        return {
            'metrics': metrics,
            'residual_statistics': residual_stats,
            'statistical_tests': statistical_tests,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'predictions': predictions,
            'actuals': test_data
        }
        
    def compare_models(self, data: List[float], 
                      models: Dict[str, Any]) -> Dict[str, Any]:
        """
        複数モデル比較
        
        Args:
            data: 時系列データ
            models: {'model_name': model_instance}形式の辞書
            
        Returns:
            比較結果
        """
        logger.info(f"モデル比較開始: {list(models.keys())}")
        
        comparison_results = {}
        model_metrics = {}
        
        # 各モデル評価
        for model_name, model in models.items():
            try:
                logger.info(f"{model_name}評価中...")
                result = self.evaluate_model(data, model, ['simple_split'])
                comparison_results[model_name] = result
                
                # 主要メトリクス抽出
                if 'simple_split' in result and 'metrics' in result['simple_split']:
                    model_metrics[model_name] = result['simple_split']['metrics']
                    
            except Exception as e:
                logger.warning(f"{model_name}評価失敗: {e}")
                comparison_results[model_name] = {'error': str(e)}
                
        # モデルランキング
        rankings = self._rank_models(model_metrics)
        
        # 統計的有意性検定
        significance_tests = self._model_significance_tests(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'model_rankings': rankings,
            'significance_tests': significance_tests,
            'summary': self._generate_comparison_summary(model_metrics, rankings)
        }
        
    def _rank_models(self, model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """モデルランキング"""
        if not model_metrics:
            return {}
            
        # 各指標でのランキング
        metrics_to_rank = ['rmse', 'mae', 'mape', 'theil_u']  # 小さいほど良い
        inverse_metrics = ['r2', 'direction_accuracy']  # 大きいほど良い
        
        rankings = {}
        
        for metric in metrics_to_rank:
            values = [(name, metrics.get(metric, float('inf'))) for name, metrics in model_metrics.items()]
            values.sort(key=lambda x: x[1])  # 昇順
            rankings[metric] = [name for name, _ in values]
            
        for metric in inverse_metrics:
            values = [(name, metrics.get(metric, -float('inf'))) for name, metrics in model_metrics.items()]
            values.sort(key=lambda x: x[1], reverse=True)  # 降順
            rankings[metric] = [name for name, _ in values]
            
        # 総合ランキング（平均順位）
        model_names = list(model_metrics.keys())
        avg_ranks = {}
        
        for model_name in model_names:
            ranks = []
            for metric, ranking in rankings.items():
                if model_name in ranking:
                    ranks.append(ranking.index(model_name) + 1)
            avg_ranks[model_name] = np.mean(ranks) if ranks else float('inf')
            
        overall_ranking = sorted(model_names, key=lambda x: avg_ranks[x])
        
        return {
            'by_metric': rankings,
            'overall': overall_ranking,
            'average_ranks': avg_ranks
        }
        
    def _model_significance_tests(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """モデル間有意性検定"""
        significance_tests = {}
        
        # 予測値抽出
        model_predictions = {}
        actuals = None
        
        for model_name, result in comparison_results.items():
            if ('simple_split' in result and 'predictions' in result['simple_split'] 
                and 'actuals' in result['simple_split']):
                model_predictions[model_name] = result['simple_split']['predictions']
                if actuals is None:
                    actuals = result['simple_split']['actuals']
                    
        if len(model_predictions) < 2 or actuals is None:
            return {'error': 'insufficient_models_for_testing'}
            
        # ペアワイズ比較
        model_names = list(model_predictions.keys())
        pairwise_tests = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                try:
                    pred1 = np.array(model_predictions[model1])
                    pred2 = np.array(model_predictions[model2])
                    actual_array = np.array(actuals)
                    
                    # 最小長に合わせる
                    min_len = min(len(pred1), len(pred2), len(actual_array))
                    pred1 = pred1[:min_len]
                    pred2 = pred2[:min_len]
                    actual_array = actual_array[:min_len]
                    
                    # 残差計算
                    residuals1 = actual_array - pred1
                    residuals2 = actual_array - pred2
                    
                    # Wilcoxon符号順位検定（対応ありt検定の代替）
                    if len(residuals1) >= 6:
                        stat, p_value = stats.wilcoxon(np.abs(residuals1), np.abs(residuals2))
                        test_result = {
                            'test': 'Wilcoxon Signed-Rank',
                            'statistic': stat,
                            'p_value': p_value,
                            'significant': p_value < self.significance_level,
                            'better_model': model1 if np.mean(np.abs(residuals1)) < np.mean(np.abs(residuals2)) else model2
                        }
                    else:
                        test_result = {'error': 'insufficient_data_for_test'}
                        
                    pairwise_tests[f"{model1}_vs_{model2}"] = test_result
                    
                except Exception as e:
                    pairwise_tests[f"{model1}_vs_{model2}"] = {'error': str(e)}
                    
        return {
            'pairwise_tests': pairwise_tests,
            'significance_level': self.significance_level
        }
        
    def _generate_comparison_summary(self, model_metrics: Dict[str, Dict[str, float]], 
                                   rankings: Dict[str, Any]) -> Dict[str, Any]:
        """比較サマリー生成"""
        if not model_metrics or not rankings:
            return {}
            
        best_model = rankings.get('overall', [None])[0]
        
        summary = {
            'best_overall_model': best_model,
            'total_models_compared': len(model_metrics)
        }
        
        if best_model and best_model in model_metrics:
            summary['best_model_metrics'] = model_metrics[best_model]
            
        # 各指標でのベストモデル
        best_by_metric = {}
        if 'by_metric' in rankings:
            for metric, ranking in rankings['by_metric'].items():
                if ranking:
                    best_by_metric[metric] = ranking[0]
        summary['best_by_metric'] = best_by_metric
        
        return summary
        
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """評価サマリー"""
        return {
            'test_ratio': self.test_ratio,
            'rolling_window': self.rolling_window,
            'significance_level': self.significance_level,
            'available_methods': ['simple_split', 'rolling_window', 'time_series_cv'],
            'evaluation_results': self.evaluation_results
        }