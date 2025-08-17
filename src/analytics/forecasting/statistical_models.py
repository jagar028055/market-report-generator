"""
統計予測モデル実装

ARIMA（AutoRegressive Integrated Moving Average）とSARIMA（Seasonal ARIMA）モデルを
提供し、時系列データの予測を行う。自動パラメータ選択、信頼区間計算、
診断統計を含む包括的な統計予測機能を実装。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class ARIMAModel:
    """
    ARIMA（AutoRegressive Integrated Moving Average）モデル
    
    時系列データの自己回帰、差分、移動平均成分を組み合わせて予測を行う。
    自動パラメータ選択（AIC基準）、信頼区間計算、診断統計を提供。
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        """
        Args:
            max_p: 自己回帰次数の最大値
            max_d: 差分次数の最大値  
            max_q: 移動平均次数の最大値
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.training_data = None
        
    def _calculate_aic(self, residuals: np.ndarray, k: int, n: int) -> float:
        """AIC（赤池情報量基準）を計算"""
        log_likelihood = -0.5 * n * np.log(2 * np.pi * np.var(residuals))
        log_likelihood -= 0.5 * np.sum(residuals**2) / np.var(residuals)
        return 2 * k - 2 * log_likelihood
        
    def _fit_arima(self, data: np.ndarray, p: int, d: int, q: int) -> Tuple[Optional[Any], float]:
        """
        指定されたパラメータでARIMAモデルを学習
        
        Returns:
            (fitted_model, aic_score)
        """
        try:
            # 差分処理
            diff_data = data.copy()
            for _ in range(d):
                diff_data = np.diff(diff_data)
                
            n = len(diff_data)
            if n < max(p, q) + 1:
                return None, float('inf')
                
            # AR成分の係数推定（最小二乗法）
            if p > 0:
                X_ar = np.column_stack([diff_data[p-i-1:n-i-1] for i in range(p)])
                y_ar = diff_data[p:]
                
                if len(y_ar) == 0:
                    return None, float('inf')
                    
                ar_coefs = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
            else:
                ar_coefs = np.array([])
                
            # MA成分の係数推定（簡易実装）
            if q > 0:
                # 残差を使用してMA係数を推定
                if p > 0:
                    residuals = y_ar - X_ar @ ar_coefs
                else:
                    residuals = diff_data
                    
                # 簡易MA係数推定
                ma_coefs = []
                for i in range(min(q, len(residuals) - 1)):
                    if len(residuals) > i + 1:
                        coef = np.corrcoef(residuals[:-i-1], residuals[i+1:])[0, 1]
                        ma_coefs.append(coef if not np.isnan(coef) else 0.0)
                    else:
                        ma_coefs.append(0.0)
                ma_coefs = np.array(ma_coefs)
            else:
                ma_coefs = np.array([])
                
            # 予測誤差計算
            predicted = self._predict_arima(diff_data, ar_coefs, ma_coefs, p, q)
            residuals = diff_data[max(p, q):] - predicted
            
            if len(residuals) == 0:
                return None, float('inf')
                
            # AIC計算
            k = p + q + 1  # パラメータ数
            aic = self._calculate_aic(residuals, k, len(residuals))
            
            model_info = {
                'ar_coefs': ar_coefs,
                'ma_coefs': ma_coefs,
                'p': p, 'd': d, 'q': q,
                'residuals': residuals,
                'aic': aic
            }
            
            return model_info, aic
            
        except Exception as e:
            logger.warning(f"ARIMA({p},{d},{q})の学習に失敗: {e}")
            return None, float('inf')
            
    def _predict_arima(self, data: np.ndarray, ar_coefs: np.ndarray, 
                      ma_coefs: np.ndarray, p: int, q: int) -> np.ndarray:
        """ARIMAモデルによる予測"""
        n = len(data)
        start_idx = max(p, q)
        predictions = []
        
        for i in range(start_idx, n):
            pred = 0.0
            
            # AR成分
            if p > 0 and len(ar_coefs) > 0:
                ar_values = data[i-p:i][::-1]  # 逆順
                pred += np.dot(ar_coefs[:len(ar_values)], ar_values)
                
            # MA成分（簡易実装）
            if q > 0 and len(ma_coefs) > 0:
                if i >= q:
                    ma_values = data[i-q:i][::-1]  # 逆順
                    pred += np.dot(ma_coefs[:len(ma_values)], ma_values) * 0.1
                    
            predictions.append(pred)
            
        return np.array(predictions)
        
    def auto_arima(self, data: List[float]) -> Dict[str, Any]:
        """
        自動ARIMA（AIC基準でのパラメータ選択）
        
        Args:
            data: 時系列データ
            
        Returns:
            最適モデル情報
        """
        data_array = np.array(data)
        self.training_data = data_array
        
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        logger.info(f"ARIMA自動選択開始: データ数={len(data)}")
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                        
                    model, aic = self._fit_arima(data_array, p, d, q)
                    
                    if model is not None and aic < best_aic:
                        best_aic = aic
                        best_model = model
                        best_params = (p, d, q)
                        
        if best_model is None:
            raise ValueError("ARIMAモデルの学習に失敗しました")
            
        self.fitted_model = best_model
        self.best_params = best_params
        
        logger.info(f"最適ARIMA{best_params}選択完了: AIC={best_aic:.2f}")
        
        return {
            'params': best_params,
            'aic': best_aic,
            'model': best_model
        }
        
    def forecast(self, steps: int = 10) -> Dict[str, Any]:
        """
        将来予測実行
        
        Args:
            steps: 予測ステップ数
            
        Returns:
            予測結果（値、信頼区間）
        """
        if self.fitted_model is None:
            raise ValueError("モデルが学習されていません")
            
        model = self.fitted_model
        ar_coefs = model['ar_coefs']
        ma_coefs = model['ma_coefs']
        p, d, q = model['p'], model['d'], model['q']
        
        # 差分処理されたデータの準備
        diff_data = self.training_data.copy()
        for _ in range(d):
            diff_data = np.diff(diff_data)
            
        # 予測実行
        forecasts = []
        data_extended = list(diff_data)
        
        for step in range(steps):
            pred = 0.0
            
            # AR成分
            if p > 0 and len(ar_coefs) > 0:
                ar_values = data_extended[-p:][::-1]  # 最新p個の値を逆順
                pred += np.dot(ar_coefs[:len(ar_values)], ar_values)
                
            # MA成分（簡易実装）
            if q > 0 and len(ma_coefs) > 0:
                ma_values = data_extended[-q:][::-1]  # 最新q個の値を逆順  
                pred += np.dot(ma_coefs[:len(ma_values)], ma_values) * 0.1
                
            forecasts.append(pred)
            data_extended.append(pred)
            
        # 差分の逆変換
        forecasts_integrated = forecasts.copy()
        for _ in range(d):
            # 累積和で逆変換
            if len(self.training_data) > 0:
                last_value = self.training_data[-1]
                for i in range(len(forecasts_integrated)):
                    if i == 0:
                        forecasts_integrated[i] += last_value
                    else:
                        forecasts_integrated[i] += forecasts_integrated[i-1]
                        
        # 信頼区間計算（残差の標準偏差を使用）
        residual_std = np.std(model['residuals'])
        confidence_95 = 1.96 * residual_std
        
        upper_bound = np.array(forecasts_integrated) + confidence_95
        lower_bound = np.array(forecasts_integrated) - confidence_95
        
        return {
            'forecast': forecasts_integrated,
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'confidence_interval': 95,
            'residual_std': residual_std
        }
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """モデル診断統計"""
        if self.fitted_model is None:
            raise ValueError("モデルが学習されていません")
            
        residuals = self.fitted_model['residuals']
        
        # 正規性検定（Shapiro-Wilk検定）
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
            
        # 自己相関検定（Ljung-Box検定の簡易版）
        autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        
        return {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'normality_test': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else False
            },
            'autocorrelation_lag1': autocorr_lag1,
            'aic': self.fitted_model['aic'],
            'params': self.best_params
        }


class SARIMAModel:
    """
    SARIMA（Seasonal ARIMA）モデル
    
    季節性のある時系列データに対してARIMAモデルを拡張。
    季節成分（P, D, Q, S）を含む包括的な季節調整予測を提供。
    """
    
    def __init__(self, seasonal_period: int = 12, max_p: int = 2, max_d: int = 1, 
                 max_q: int = 2, max_P: int = 1, max_D: int = 1, max_Q: int = 1):
        """
        Args:
            seasonal_period: 季節周期（月次データなら12）
            max_p, max_d, max_q: 非季節ARIMAパラメータの最大値
            max_P, max_D, max_Q: 季節ARIMAパラメータの最大値
        """
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.fitted_model = None
        self.best_params = None
        self.training_data = None
        
    def _seasonal_decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """季節分解（加法モデル）"""
        n = len(data)
        s = self.seasonal_period
        
        if n < 2 * s:
            # データが不足している場合は簡易分解
            trend = np.full(n, np.mean(data))
            seasonal = np.zeros(n)
            residual = data - trend
            return {'trend': trend, 'seasonal': seasonal, 'residual': residual}
            
        # トレンド成分（移動平均）
        trend = np.full(n, np.nan)
        for i in range(s//2, n - s//2):
            trend[i] = np.mean(data[i-s//2:i+s//2+1])
            
        # 前後の値で補間
        trend[:s//2] = trend[s//2]
        trend[n-s//2:] = trend[n-s//2-1]
        
        # 季節成分
        detrended = data - trend
        seasonal = np.zeros(n)
        
        for i in range(s):
            season_values = detrended[i::s]
            season_values = season_values[~np.isnan(season_values)]
            if len(season_values) > 0:
                seasonal[i::s] = np.mean(season_values)
                
        # 残差成分
        residual = data - trend - seasonal
        
        return {'trend': trend, 'seasonal': seasonal, 'residual': residual}
        
    def _fit_sarima(self, data: np.ndarray, p: int, d: int, q: int, 
                   P: int, D: int, Q: int) -> Tuple[Optional[Any], float]:
        """SARIMA(p,d,q)(P,D,Q,s)モデルの学習"""
        try:
            s = self.seasonal_period
            
            # 通常差分
            diff_data = data.copy()
            for _ in range(d):
                diff_data = np.diff(diff_data)
                
            # 季節差分
            for _ in range(D):
                if len(diff_data) > s:
                    diff_data = diff_data[s:] - diff_data[:-s]
                else:
                    break
                    
            n = len(diff_data)
            if n < max(p, q, P*s, Q*s) + 1:
                return None, float('inf')
                
            # 非季節AR成分
            X_components = []
            if p > 0:
                for i in range(p):
                    if i < n - max(p, q, P*s, Q*s):
                        X_components.append(diff_data[p-i-1:n-max(p, q, P*s, Q*s)+p-i-1])
                        
            # 季節AR成分
            if P > 0:
                for i in range(P):
                    lag = (i + 1) * s
                    if lag < n - max(p, q, P*s, Q*s):
                        X_components.append(diff_data[lag-max(p, q, P*s, Q*s):n-max(p, q, P*s, Q*s)+lag])
                        
            if X_components:
                X = np.column_stack(X_components)
                y = diff_data[max(p, q, P*s, Q*s):]
                
                if len(y) == 0:
                    return None, float('inf')
                    
                coefs = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ coefs
            else:
                coefs = np.array([])
                residuals = diff_data
                
            if len(residuals) == 0:
                return None, float('inf')
                
            # AIC計算
            k = p + q + P + Q + 1
            aic = self._calculate_aic(residuals, k, len(residuals))
            
            model_info = {
                'coefs': coefs,
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q,
                's': s,
                'residuals': residuals,
                'aic': aic,
                'diff_data': diff_data
            }
            
            return model_info, aic
            
        except Exception as e:
            logger.warning(f"SARIMA({p},{d},{q})({P},{D},{Q},{s})の学習に失敗: {e}")
            return None, float('inf')
            
    def _calculate_aic(self, residuals: np.ndarray, k: int, n: int) -> float:
        """AIC計算"""
        log_likelihood = -0.5 * n * np.log(2 * np.pi * np.var(residuals))
        log_likelihood -= 0.5 * np.sum(residuals**2) / np.var(residuals)
        return 2 * k - 2 * log_likelihood
        
    def auto_sarima(self, data: List[float]) -> Dict[str, Any]:
        """
        自動SARIMA（AIC基準でのパラメータ選択）
        
        Args:
            data: 時系列データ
            
        Returns:
            最適モデル情報
        """
        data_array = np.array(data)
        self.training_data = data_array
        
        # 季節分解による前処理分析
        decomposition = self._seasonal_decompose(data_array)
        seasonal_strength = np.std(decomposition['seasonal']) / np.std(data_array)
        
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        logger.info(f"SARIMA自動選択開始: データ数={len(data)}, 季節性強度={seasonal_strength:.3f}")
        
        # パラメータ探索
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    for P in range(self.max_P + 1):
                        for D in range(self.max_D + 1):
                            for Q in range(self.max_Q + 1):
                                if p + d + q + P + D + Q == 0:
                                    continue
                                    
                                model, aic = self._fit_sarima(data_array, p, d, q, P, D, Q)
                                
                                if model is not None and aic < best_aic:
                                    best_aic = aic
                                    best_model = model
                                    best_params = (p, d, q, P, D, Q)
                                    
        if best_model is None:
            # フォールバック: 簡単なSARIMAモデル
            logger.warning("最適SARIMAが見つからないため簡易モデルを使用")
            model, aic = self._fit_sarima(data_array, 1, 1, 1, 1, 1, 1)
            if model is not None:
                best_model = model
                best_params = (1, 1, 1, 1, 1, 1)
                best_aic = aic
            else:
                raise ValueError("SARIMAモデルの学習に失敗しました")
                
        self.fitted_model = best_model
        self.best_params = best_params
        
        logger.info(f"最適SARIMA{best_params}選択完了: AIC={best_aic:.2f}")
        
        return {
            'params': best_params,
            'aic': best_aic,
            'seasonal_strength': seasonal_strength,
            'model': best_model
        }
        
    def forecast(self, steps: int = 10) -> Dict[str, Any]:
        """
        季節性を考慮した将来予測
        
        Args:
            steps: 予測ステップ数
            
        Returns:
            予測結果（値、信頼区間、季節成分）
        """
        if self.fitted_model is None:
            raise ValueError("モデルが学習されていません")
            
        model = self.fitted_model
        p, d, q = model['p'], model['d'], model['q']
        P, D, Q, s = model['P'], model['D'], model['Q'], model['s']
        
        # 簡易予測（トレンド + 季節パターン）
        decomposition = self._seasonal_decompose(self.training_data)
        recent_trend = np.mean(decomposition['trend'][-min(12, len(decomposition['trend'])):])
        seasonal_pattern = decomposition['seasonal'][-s:] if len(decomposition['seasonal']) >= s else np.zeros(s)
        
        forecasts = []
        for step in range(steps):
            # 基本トレンド
            forecast_value = recent_trend
            
            # 季節成分追加
            seasonal_idx = step % s
            if seasonal_idx < len(seasonal_pattern):
                forecast_value += seasonal_pattern[seasonal_idx]
                
            forecasts.append(forecast_value)
            
        # 信頼区間
        residual_std = np.std(model['residuals'])
        confidence_95 = 1.96 * residual_std
        
        upper_bound = np.array(forecasts) + confidence_95
        lower_bound = np.array(forecasts) - confidence_95
        
        return {
            'forecast': forecasts,
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'seasonal_component': [seasonal_pattern[i % s] for i in range(steps)],
            'trend_component': [recent_trend] * steps,
            'confidence_interval': 95,
            'residual_std': residual_std
        }
        
    def get_seasonal_diagnostics(self) -> Dict[str, Any]:
        """季節性診断"""
        if self.training_data is None:
            raise ValueError("データが設定されていません")
            
        decomposition = self._seasonal_decompose(self.training_data)
        
        # 季節性強度
        seasonal_strength = np.std(decomposition['seasonal']) / np.std(self.training_data)
        
        # 季節パターンの安定性
        s = self.seasonal_period
        n_seasons = len(self.training_data) // s
        
        seasonal_stability = 0.0
        if n_seasons > 1:
            seasonal_patterns = []
            for i in range(n_seasons):
                start_idx = i * s
                end_idx = min((i + 1) * s, len(self.training_data))
                season_data = self.training_data[start_idx:end_idx]
                if len(season_data) == s:
                    seasonal_patterns.append(season_data)
                    
            if len(seasonal_patterns) > 1:
                correlations = []
                for i in range(len(seasonal_patterns) - 1):
                    corr = np.corrcoef(seasonal_patterns[i], seasonal_patterns[i+1])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                if correlations:
                    seasonal_stability = np.mean(correlations)
                    
        return {
            'seasonal_strength': seasonal_strength,
            'seasonal_stability': seasonal_stability,
            'seasonal_period': s,
            'trend_strength': np.std(decomposition['trend']) / np.std(self.training_data),
            'residual_strength': np.std(decomposition['residual']) / np.std(self.training_data),
            'model_params': self.best_params
        }