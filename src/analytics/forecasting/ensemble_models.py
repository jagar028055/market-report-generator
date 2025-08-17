"""
アンサンブル予測モデル実装

複数予測モデルを組み合わせて予測精度を向上させるアンサンブル手法。
重み付け平均、スタッキング、動的重み調整、予測信頼度評価を含む
包括的なアンサンブル予測システム。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from .statistical_models import ARIMAModel, SARIMAModel
from .ml_models import RandomForestModel, XGBoostModel

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    アンサンブル予測システム
    
    ARIMA、SARIMA、Random Forest、XGBoostを組み合わせた予測。
    動的重み調整、予測信頼度評価、モデル相関分析を提供。
    """
    
    def __init__(self, ensemble_method: str = 'weighted_average',
                 optimize_weights: bool = True, confidence_threshold: float = 0.8):
        """
        Args:
            ensemble_method: アンサンブル手法 ('weighted_average', 'stacking')
            optimize_weights: 重み最適化フラグ
            confidence_threshold: 予測信頼度閾値
        """
        self.ensemble_method = ensemble_method
        self.optimize_weights = optimize_weights
        self.confidence_threshold = confidence_threshold
        
        # モデル
        self.arima_model = ARIMAModel()
        self.sarima_model = SARIMAModel()
        self.rf_model = RandomForestModel()
        self.xgb_model = XGBoostModel()
        
        # アンサンブル情報
        self.model_weights = {}
        self.model_performance = {}
        self.stacking_model = None
        self.is_fitted = False
        self.training_data = None
        
    def _validate_data(self, data: List[float]) -> np.ndarray:
        """データ検証"""
        if len(data) < 20:
            raise ValueError(f"データ数が不足しています: {len(data)} (最低20必要)")
            
        data_array = np.array(data)
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            raise ValueError("データに無効値が含まれています")
            
        return data_array
        
    def _fit_individual_models(self, data: List[float]) -> Dict[str, Dict[str, Any]]:
        """個別モデルの学習"""
        results = {}
        
        # ARIMA
        try:
            logger.info("ARIMA学習開始")
            arima_result = self.arima_model.auto_arima(data)
            results['arima'] = {
                'success': True,
                'result': arima_result,
                'model': self.arima_model
            }
        except Exception as e:
            logger.warning(f"ARIMA学習失敗: {e}")
            results['arima'] = {'success': False, 'error': str(e)}
            
        # SARIMA
        try:
            logger.info("SARIMA学習開始")
            sarima_result = self.sarima_model.auto_sarima(data)
            results['sarima'] = {
                'success': True,
                'result': sarima_result,
                'model': self.sarima_model
            }
        except Exception as e:
            logger.warning(f"SARIMA学習失敗: {e}")
            results['sarima'] = {'success': False, 'error': str(e)}
            
        # Random Forest
        try:
            logger.info("Random Forest学習開始")
            rf_result = self.rf_model.fit(data)
            results['random_forest'] = {
                'success': True,
                'result': rf_result,
                'model': self.rf_model
            }
        except Exception as e:
            logger.warning(f"Random Forest学習失敗: {e}")
            results['random_forest'] = {'success': False, 'error': str(e)}
            
        # XGBoost
        try:
            logger.info("XGBoost学習開始")
            xgb_result = self.xgb_model.fit(data)
            results['xgboost'] = {
                'success': True,
                'result': xgb_result,
                'model': self.xgb_model
            }
        except Exception as e:
            logger.warning(f"XGBoost学習失敗: {e}")
            results['xgboost'] = {'success': False, 'error': str(e)}
            
        return results
        
    def _calculate_model_performance(self, data: List[float], 
                                   fitted_models: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """モデル性能評価（バックテスト）"""
        performance = {}
        
        # 学習・検証データ分割（時系列順）
        split_point = int(len(data) * 0.8)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        if len(test_data) < 5:
            # テストデータが少ない場合は学習データ性能を使用
            for model_name, model_info in fitted_models.items():
                if model_info['success']:
                    if 'training_metrics' in model_info['result']:
                        performance[model_name] = 1.0 / (1.0 + model_info['result']['training_metrics'].get('mse', 1.0))
                    else:
                        performance[model_name] = 0.5  # デフォルト
                else:
                    performance[model_name] = 0.0
            return performance
            
        # 各モデルの予測性能評価
        for model_name, model_info in fitted_models.items():
            if not model_info['success']:
                performance[model_name] = 0.0
                continue
                
            try:
                model = model_info['model']
                
                if model_name == 'arima':
                    forecast_result = model.forecast(len(test_data))
                    predictions = forecast_result['forecast']
                elif model_name == 'sarima':
                    forecast_result = model.forecast(len(test_data))
                    predictions = forecast_result['forecast']
                elif model_name in ['random_forest', 'xgboost']:
                    predict_result = model.predict(train_data, len(test_data))
                    predictions = predict_result['predictions']
                else:
                    continue
                    
                # 予測数調整
                predictions = predictions[:len(test_data)]
                if len(predictions) < len(test_data):
                    # 足りない場合は最後の値で埋める
                    last_pred = predictions[-1] if predictions else test_data[0]
                    predictions.extend([last_pred] * (len(test_data) - len(predictions)))
                    
                # MSE計算（逆数で性能スコア化）
                mse = mean_squared_error(test_data, predictions)
                performance[model_name] = 1.0 / (1.0 + mse)
                
            except Exception as e:
                logger.warning(f"{model_name}の性能評価失敗: {e}")
                performance[model_name] = 0.0
                
        return performance
        
    def _optimize_ensemble_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        """アンサンブル重みの最適化"""
        if not self.optimize_weights:
            # 均等重み
            successful_models = [name for name, perf in performance.items() if perf > 0]
            if not successful_models:
                return {}
            weight_value = 1.0 / len(successful_models)
            return {name: weight_value for name in successful_models}
            
        # 性能ベース重み計算
        total_performance = sum(performance.values())
        if total_performance == 0:
            return {}
            
        # ソフトマックス重み
        weights = {}
        for model_name, perf in performance.items():
            if perf > 0:
                weights[model_name] = perf / total_performance
                
        # 重み正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
            
        return weights
        
    def _train_stacking_model(self, data: List[float], 
                            fitted_models: Dict[str, Dict[str, Any]]) -> Optional[LinearRegression]:
        """スタッキングモデルの学習"""
        try:
            # 各モデルの予測値を特徴量として使用
            split_point = int(len(data) * 0.7)
            train_data = data[:split_point]
            meta_train_data = data[split_point:int(len(data) * 0.85)]
            
            if len(meta_train_data) < 5:
                return None
                
            meta_features = []
            
            for model_name, model_info in fitted_models.items():
                if not model_info['success']:
                    continue
                    
                try:
                    model = model_info['model']
                    
                    if model_name == 'arima':
                        forecast_result = model.forecast(len(meta_train_data))
                        predictions = forecast_result['forecast']
                    elif model_name == 'sarima':
                        forecast_result = model.forecast(len(meta_train_data))
                        predictions = forecast_result['forecast']
                    elif model_name in ['random_forest', 'xgboost']:
                        predict_result = model.predict(train_data, len(meta_train_data))
                        predictions = predict_result['predictions']
                    else:
                        continue
                        
                    # 予測数調整
                    predictions = predictions[:len(meta_train_data)]
                    if len(predictions) == len(meta_train_data):
                        meta_features.append(predictions)
                        
                except Exception as e:
                    logger.warning(f"スタッキング特徴量生成失敗({model_name}): {e}")
                    continue
                    
            if len(meta_features) < 2:
                return None
                
            # メタ学習
            X_meta = np.column_stack(meta_features)
            y_meta = np.array(meta_train_data)
            
            stacking_model = LinearRegression()
            stacking_model.fit(X_meta, y_meta)
            
            return stacking_model
            
        except Exception as e:
            logger.warning(f"スタッキングモデル学習失敗: {e}")
            return None
            
    def fit(self, data: List[float]) -> Dict[str, Any]:
        """
        アンサンブルモデル学習
        
        Args:
            data: 時系列データ
            
        Returns:
            学習結果とモデル情報
        """
        logger.info(f"アンサンブル学習開始: データ数={len(data)}")
        
        # データ検証
        self.training_data = self._validate_data(data)
        
        # 個別モデル学習
        fitted_models = self._fit_individual_models(data)
        
        # 成功モデル確認
        successful_models = [name for name, info in fitted_models.items() if info['success']]
        if not successful_models:
            raise ValueError("全モデルの学習に失敗しました")
            
        logger.info(f"学習成功モデル: {successful_models}")
        
        # モデル性能評価
        self.model_performance = self._calculate_model_performance(data, fitted_models)
        
        # 重み最適化
        self.model_weights = self._optimize_ensemble_weights(self.model_performance)
        
        # スタッキングモデル学習
        if self.ensemble_method == 'stacking':
            self.stacking_model = self._train_stacking_model(data, fitted_models)
            
        self.is_fitted = True
        
        # 結果サマリー
        ensemble_summary = {
            'successful_models': successful_models,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'ensemble_method': self.ensemble_method,
            'stacking_available': self.stacking_model is not None
        }
        
        logger.info(f"アンサンブル学習完了: 重み={self.model_weights}")
        
        return ensemble_summary
        
    def forecast(self, steps: int = 10) -> Dict[str, Any]:
        """
        アンサンブル予測
        
        Args:
            steps: 予測ステップ数
            
        Returns:
            予測結果と信頼度情報
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
            
        logger.info(f"アンサンブル予測開始: {steps}ステップ")
        
        # 個別モデル予測
        individual_predictions = {}
        model_confidences = {}
        
        # ARIMA
        if 'arima' in self.model_weights and self.arima_model.fitted_model is not None:
            try:
                result = self.arima_model.forecast(steps)
                individual_predictions['arima'] = result['forecast']
                model_confidences['arima'] = 1.0 / (1.0 + result.get('residual_std', 1.0))
            except Exception as e:
                logger.warning(f"ARIMA予測失敗: {e}")
                
        # SARIMA
        if 'sarima' in self.model_weights and self.sarima_model.fitted_model is not None:
            try:
                result = self.sarima_model.forecast(steps)
                individual_predictions['sarima'] = result['forecast']
                model_confidences['sarima'] = 1.0 / (1.0 + result.get('residual_std', 1.0))
            except Exception as e:
                logger.warning(f"SARIMA予測失敗: {e}")
                
        # Random Forest
        if 'random_forest' in self.model_weights and self.rf_model.is_fitted:
            try:
                result = self.rf_model.predict(self.training_data.tolist(), steps)
                individual_predictions['random_forest'] = result['predictions']
                model_confidences['random_forest'] = self.model_performance.get('random_forest', 0.5)
            except Exception as e:
                logger.warning(f"Random Forest予測失敗: {e}")
                
        # XGBoost
        if 'xgboost' in self.model_weights and self.xgb_model.is_fitted:
            try:
                result = self.xgb_model.predict(self.training_data.tolist(), steps)
                individual_predictions['xgboost'] = result['predictions']
                model_confidences['xgboost'] = self.model_performance.get('xgboost', 0.5)
            except Exception as e:
                logger.warning(f"XGBoost予測失敗: {e}")
                
        if not individual_predictions:
            raise ValueError("全モデルの予測に失敗しました")
            
        # アンサンブル予測実行
        if self.ensemble_method == 'weighted_average':
            ensemble_forecast = self._weighted_average_forecast(individual_predictions)
        elif self.ensemble_method == 'stacking' and self.stacking_model is not None:
            ensemble_forecast = self._stacking_forecast(individual_predictions)
        else:
            # フォールバック: 単純平均
            ensemble_forecast = self._simple_average_forecast(individual_predictions)
            
        # 信頼区間計算
        confidence_info = self._calculate_ensemble_confidence(
            individual_predictions, model_confidences, ensemble_forecast
        )
        
        # 予測分析
        prediction_analysis = self._analyze_predictions(individual_predictions, ensemble_forecast)
        
        return {
            'ensemble_forecast': ensemble_forecast,
            'individual_predictions': individual_predictions,
            'confidence_interval': confidence_info,
            'prediction_analysis': prediction_analysis,
            'model_weights': self.model_weights,
            'forecast_confidence': np.mean(list(model_confidences.values()))
        }
        
    def _weighted_average_forecast(self, predictions: Dict[str, List[float]]) -> List[float]:
        """重み付き平均予測"""
        steps = len(next(iter(predictions.values())))
        ensemble_forecast = []
        
        for step in range(steps):
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, preds in predictions.items():
                if model_name in self.model_weights and step < len(preds):
                    weight = self.model_weights[model_name]
                    weighted_sum += weight * preds[step]
                    total_weight += weight
                    
            if total_weight > 0:
                ensemble_forecast.append(weighted_sum / total_weight)
            else:
                # フォールバック
                avg_pred = np.mean([preds[step] for preds in predictions.values() if step < len(preds)])
                ensemble_forecast.append(avg_pred)
                
        return ensemble_forecast
        
    def _stacking_forecast(self, predictions: Dict[str, List[float]]) -> List[float]:
        """スタッキング予測"""
        if self.stacking_model is None:
            return self._simple_average_forecast(predictions)
            
        try:
            steps = len(next(iter(predictions.values())))
            ensemble_forecast = []
            
            # モデル名順序統一
            model_names = sorted(predictions.keys())
            
            for step in range(steps):
                step_features = []
                for model_name in model_names:
                    if step < len(predictions[model_name]):
                        step_features.append(predictions[model_name][step])
                    else:
                        step_features.append(0.0)  # フォールバック
                        
                if len(step_features) == len(model_names):
                    X_meta = np.array(step_features).reshape(1, -1)
                    pred = self.stacking_model.predict(X_meta)[0]
                    ensemble_forecast.append(pred)
                else:
                    # フォールバック
                    avg_pred = np.mean([preds[step] for preds in predictions.values() if step < len(preds)])
                    ensemble_forecast.append(avg_pred)
                    
            return ensemble_forecast
            
        except Exception as e:
            logger.warning(f"スタッキング予測失敗: {e}")
            return self._simple_average_forecast(predictions)
            
    def _simple_average_forecast(self, predictions: Dict[str, List[float]]) -> List[float]:
        """単純平均予測"""
        steps = len(next(iter(predictions.values())))
        ensemble_forecast = []
        
        for step in range(steps):
            step_predictions = [preds[step] for preds in predictions.values() if step < len(preds)]
            if step_predictions:
                ensemble_forecast.append(np.mean(step_predictions))
            else:
                ensemble_forecast.append(0.0)
                
        return ensemble_forecast
        
    def _calculate_ensemble_confidence(self, predictions: Dict[str, List[float]], 
                                     confidences: Dict[str, float],
                                     ensemble_forecast: List[float]) -> Dict[str, Any]:
        """アンサンブル信頼区間計算"""
        steps = len(ensemble_forecast)
        
        # 予測値分散
        prediction_variance = []
        for step in range(steps):
            step_preds = [preds[step] for preds in predictions.values() if step < len(preds)]
            if len(step_preds) > 1:
                prediction_variance.append(np.var(step_preds))
            else:
                prediction_variance.append(0.1)  # デフォルト分散
                
        # 重み付き信頼度
        weighted_confidence = 0.0
        total_weight = 0.0
        for model_name, conf in confidences.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                weighted_confidence += weight * conf
                total_weight += weight
                
        if total_weight > 0:
            weighted_confidence /= total_weight
        else:
            weighted_confidence = 0.5
            
        # 信頼区間計算
        confidence_multiplier = 1.96  # 95%信頼区間
        std_errors = np.sqrt(prediction_variance)
        
        upper_bound = np.array(ensemble_forecast) + confidence_multiplier * std_errors
        lower_bound = np.array(ensemble_forecast) - confidence_multiplier * std_errors
        
        return {
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'confidence_level': 95,
            'weighted_confidence': weighted_confidence,
            'prediction_variance': prediction_variance
        }
        
    def _analyze_predictions(self, individual_predictions: Dict[str, List[float]], 
                           ensemble_forecast: List[float]) -> Dict[str, Any]:
        """予測分析"""
        # モデル間相関
        model_correlations = {}
        model_names = list(individual_predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                try:
                    corr = np.corrcoef(individual_predictions[model1], individual_predictions[model2])[0, 1]
                    if not np.isnan(corr):
                        model_correlations[f"{model1}_vs_{model2}"] = corr
                except:
                    model_correlations[f"{model1}_vs_{model2}"] = 0.0
                    
        # 予測多様性
        diversity_scores = []
        steps = len(ensemble_forecast)
        
        for step in range(steps):
            step_preds = [preds[step] for preds in individual_predictions.values() if step < len(preds)]
            if len(step_preds) > 1:
                diversity_scores.append(np.std(step_preds) / (np.mean(np.abs(step_preds)) + 1e-8))
            else:
                diversity_scores.append(0.0)
                
        return {
            'model_correlations': model_correlations,
            'prediction_diversity': {
                'mean': np.mean(diversity_scores),
                'std': np.std(diversity_scores),
                'max': np.max(diversity_scores),
                'min': np.min(diversity_scores)
            },
            'ensemble_vs_individual': {
                model_name: np.corrcoef(preds, ensemble_forecast[:len(preds)])[0, 1] 
                if len(preds) == len(ensemble_forecast) else 0.0
                for model_name, preds in individual_predictions.items()
            }
        }
        
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """アンサンブルサマリー"""
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")
            
        return {
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'active_models': list(self.model_weights.keys()),
            'stacking_available': self.stacking_model is not None,
            'optimization_enabled': self.optimize_weights,
            'confidence_threshold': self.confidence_threshold
        }