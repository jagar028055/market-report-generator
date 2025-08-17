"""
Volatility Analysis Module

ボラティリティ分析 - 標準偏差、VaR、GARCH系分析
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import statistics

# 依存ライブラリ（条件付きインポート）
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class VolatilityResult:
    """ボラティリティ分析結果"""
    current_volatility: float
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'
    annualized_volatility: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    metadata: Dict[str, Any]


class VolatilityAnalyzer:
    """
    ボラティリティ分析クラス
    
    Features:
    - 歴史的ボラティリティ計算
    - ローリングボラティリティ
    - VaR (Value at Risk) 計算
    - ボラティリティレジーム検出
    - GARCH系モデル（簡易版）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ボラティリティアナライザー初期化
        
        Args:
            config: 設定パラメータ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.default_window = self.config.get('volatility_window', 20)
        self.trading_days_year = self.config.get('trading_days_year', 252)
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        
        # ボラティリティレジーム閾値
        self.volatility_thresholds = {
            'low': 0.15,      # 15%未満
            'normal': 0.25,   # 15-25%
            'high': 0.40,     # 25-40%
            'extreme': float('inf')  # 40%以上
        }
        
        self.logger.debug("VolatilityAnalyzer initialized")
    
    def analyze_volatility(self, data) -> 'AnalysisResult':
        """包括的ボラティリティ分析"""
        try:
            from .analyzer import AnalysisResult
            
            # リターン計算
            returns_data = self._calculate_returns(data)
            
            if len(returns_data) == 0:
                raise ValueError("Insufficient data for volatility analysis")
            
            # 現在のボラティリティ
            current_vol = self.calculate_historical_volatility(data, window=self.default_window)
            
            # ローリングボラティリティ
            rolling_vol = self.calculate_rolling_volatility(data, window=self.default_window)
            
            # 年率化ボラティリティ
            annualized_vol = current_vol * math.sqrt(self.trading_days_year)
            
            # VaR計算
            var_results = self.calculate_var(returns_data, confidence_levels=self.confidence_levels)
            
            # ボラティリティレジーム
            regime = self.classify_volatility_regime(annualized_vol)
            
            # ボラティリティクラスタリング
            clustering = self.detect_volatility_clustering(rolling_vol)
            
            # GARCH効果検出
            garch_effect = self.detect_garch_effects(returns_data)
            
            result_data = {
                'current_volatility': current_vol,
                'annualized_volatility': annualized_vol,
                'volatility_regime': regime,
                'rolling_volatility': rolling_vol,
                'var_analysis': var_results,
                'volatility_clustering': clustering,
                'garch_effects': garch_effect,
                'volatility_statistics': self._calculate_volatility_statistics(rolling_vol.values if hasattr(rolling_vol, 'values') else [])
            }
            
            # 信頼度計算
            confidence = self._calculate_volatility_confidence(len(returns_data), current_vol)
            
            return AnalysisResult(
                analysis_type='volatility',
                result_data=result_data,
                confidence=confidence,
                metadata={
                    'returns_count': len(returns_data),
                    'window': self.default_window,
                    'trading_days_year': self.trading_days_year
                }
            )
            
        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {str(e)}")
            raise
    
    def calculate_historical_volatility(self, data, window: int = 20) -> float:
        """歴史的ボラティリティ計算"""
        try:
            returns = self._calculate_returns(data)
            
            if len(returns) < window:
                window = len(returns)
            
            if window < 2:
                return 0.0
            
            # 直近のリターンを使用
            recent_returns = returns[-window:]
            
            # ボラティリティ（標準偏差）
            volatility = statistics.stdev(recent_returns)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Historical volatility calculation failed: {str(e)}")
            return 0.0
    
    def calculate_rolling_volatility(self, data, window: int = 20):
        """ローリングボラティリティ計算"""
        try:
            from .analyzer import TimeSeriesData
            
            returns = self._calculate_returns(data)
            
            if len(returns) < window:
                self.logger.warning(f"Insufficient data for rolling volatility (need {window}, got {len(returns)})")
                return TimeSeriesData(
                    timestamps=[],
                    values=[],
                    name=f"{data.name}_rolling_volatility",
                    metadata={'window': window}
                )
            
            rolling_volatilities = []
            rolling_timestamps = []
            
            for i in range(window - 1, len(returns)):
                window_returns = returns[i - window + 1:i + 1]
                vol = statistics.stdev(window_returns) if len(window_returns) > 1 else 0.0
                rolling_volatilities.append(vol)
                # タイムスタンプはリターン計算で1つずれているので調整
                rolling_timestamps.append(data.timestamps[i + 1])
            
            return TimeSeriesData(
                timestamps=rolling_timestamps,
                values=rolling_volatilities,
                name=f"{data.name}_rolling_volatility_{window}",
                metadata={'window': window, 'type': 'rolling_volatility'}
            )
            
        except Exception as e:
            self.logger.error(f"Rolling volatility calculation failed: {str(e)}")
            raise
    
    def calculate_var(self, returns: List[float], 
                     confidence_levels: List[float] = None) -> Dict[str, Any]:
        """VaR (Value at Risk) 計算"""
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]
            
            if len(returns) == 0:
                return {'error': 'No returns data'}
            
            # 正規分布仮定のパラメトリックVaR
            mean_return = statistics.mean(returns)
            vol_return = statistics.stdev(returns) if len(returns) > 1 else 0.0
            
            parametric_var = {}
            for conf_level in confidence_levels:
                # 正規分布の分位点（標準ライブラリ実装）
                z_score = self._get_normal_quantile(conf_level)
                var_value = -(mean_return + z_score * vol_return)
                parametric_var[f'var_{int(conf_level*100)}'] = var_value
            
            # 履歴シミュレーションVaR
            sorted_returns = sorted(returns)
            historical_var = {}
            for conf_level in confidence_levels:
                percentile_index = int((1 - conf_level) * len(sorted_returns))
                percentile_index = max(0, min(percentile_index, len(sorted_returns) - 1))
                var_value = -sorted_returns[percentile_index]
                historical_var[f'var_{int(conf_level*100)}'] = var_value
            
            # 期待ショートフォール（CVaR）
            expected_shortfall = {}
            for conf_level in confidence_levels:
                percentile_index = int((1 - conf_level) * len(sorted_returns))
                tail_returns = sorted_returns[:percentile_index + 1]
                if tail_returns:
                    es_value = -statistics.mean(tail_returns)
                    expected_shortfall[f'es_{int(conf_level*100)}'] = es_value
            
            return {
                'parametric_var': parametric_var,
                'historical_var': historical_var,
                'expected_shortfall': expected_shortfall,
                'return_statistics': {
                    'mean': mean_return,
                    'volatility': vol_return,
                    'skewness': self._calculate_skewness(returns),
                    'kurtosis': self._calculate_kurtosis(returns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def classify_volatility_regime(self, annualized_volatility: float) -> str:
        """ボラティリティレジーム分類"""
        try:
            for regime, threshold in self.volatility_thresholds.items():
                if annualized_volatility < threshold:
                    return regime
            return 'extreme'
            
        except Exception:
            return 'unknown'
    
    def detect_volatility_clustering(self, rolling_volatility) -> Dict[str, Any]:
        """ボラティリティクラスタリング検出"""
        try:
            if not hasattr(rolling_volatility, 'values') or len(rolling_volatility.values) < 10:
                return {'clustering_detected': False, 'reason': 'insufficient_data'}
            
            volatilities = rolling_volatility.values
            
            # 高ボラティリティ期間の検出
            vol_mean = statistics.mean(volatilities)
            vol_std = statistics.stdev(volatilities) if len(volatilities) > 1 else 0.0
            
            high_vol_threshold = vol_mean + vol_std
            low_vol_threshold = vol_mean - vol_std
            
            # クラスター検出
            clusters = []
            current_cluster = None
            
            for i, vol in enumerate(volatilities):
                if vol > high_vol_threshold:
                    cluster_type = 'high'
                elif vol < low_vol_threshold:
                    cluster_type = 'low'
                else:
                    cluster_type = 'normal'
                
                if current_cluster is None or current_cluster['type'] != cluster_type:
                    # 新しいクラスター開始
                    if current_cluster is not None:
                        clusters.append(current_cluster)
                    
                    current_cluster = {
                        'type': cluster_type,
                        'start': i,
                        'end': i,
                        'length': 1,
                        'max_vol': vol,
                        'min_vol': vol
                    }
                else:
                    # 既存クラスター継続
                    current_cluster['end'] = i
                    current_cluster['length'] += 1
                    current_cluster['max_vol'] = max(current_cluster['max_vol'], vol)
                    current_cluster['min_vol'] = min(current_cluster['min_vol'], vol)
            
            if current_cluster is not None:
                clusters.append(current_cluster)
            
            # クラスタリング統計
            high_vol_clusters = [c for c in clusters if c['type'] == 'high' and c['length'] >= 3]
            clustering_detected = len(high_vol_clusters) > 0
            
            return {
                'clustering_detected': clustering_detected,
                'total_clusters': len(clusters),
                'high_volatility_clusters': len(high_vol_clusters),
                'cluster_details': clusters,
                'volatility_persistence': self._calculate_volatility_persistence(volatilities)
            }
            
        except Exception as e:
            self.logger.error(f"Volatility clustering detection failed: {str(e)}")
            return {'clustering_detected': False, 'error': str(e)}
    
    def detect_garch_effects(self, returns: List[float]) -> Dict[str, Any]:
        """GARCH効果検出（簡易版）"""
        try:
            if len(returns) < 20:
                return {'garch_effects': False, 'reason': 'insufficient_data'}
            
            # リターンの二乗（ボラティリティプロキシ）
            squared_returns = [r ** 2 for r in returns]
            
            # 自己相関検定（簡易版）
            lag_1_correlation = self._calculate_autocorrelation(squared_returns, lag=1)
            lag_5_correlation = self._calculate_autocorrelation(squared_returns, lag=5)
            
            # 平均回帰テスト
            mean_squared_return = statistics.mean(squared_returns)
            deviations = [sr - mean_squared_return for sr in squared_returns]
            
            # ボラティリティの永続性
            persistence = abs(lag_1_correlation)
            
            # GARCH効果の判定
            garch_threshold = 0.1  # 10%以上の自己相関
            garch_effects = abs(lag_1_correlation) > garch_threshold
            
            return {
                'garch_effects': garch_effects,
                'lag_1_autocorr': lag_1_correlation,
                'lag_5_autocorr': lag_5_correlation,
                'volatility_persistence': persistence,
                'mean_squared_return': mean_squared_return,
                'volatility_clustering_score': max(abs(lag_1_correlation), abs(lag_5_correlation))
            }
            
        except Exception as e:
            self.logger.error(f"GARCH effects detection failed: {str(e)}")
            return {'garch_effects': False, 'error': str(e)}
    
    def _calculate_returns(self, data) -> List[float]:
        """リターン計算"""
        try:
            values = data.values
            
            if len(values) < 2:
                return []
            
            returns = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    ret = (values[i] - values[i-1]) / values[i-1]
                    returns.append(ret)
                else:
                    returns.append(0.0)
            
            return returns
            
        except Exception:
            return []
    
    def _get_normal_quantile(self, confidence_level: float) -> float:
        """正規分布の分位点（近似計算）"""
        try:
            # 逆正規分布関数の近似（Beasley-Springer-Moro algorithm の簡易版）
            if confidence_level <= 0.5:
                p = confidence_level
            else:
                p = 1 - confidence_level
            
            if p <= 0:
                return -float('inf')
            if p >= 1:
                return float('inf')
            
            # 係数
            a0 = 2.515517
            a1 = 0.802853
            a2 = 0.010328
            b1 = 1.432788
            b2 = 0.189269
            b3 = 0.001308
            
            t = math.sqrt(-2 * math.log(p))
            
            numerator = a0 + a1 * t + a2 * t * t
            denominator = 1 + b1 * t + b2 * t * t + b3 * t * t * t
            
            z = t - numerator / denominator
            
            if confidence_level <= 0.5:
                return -z
            else:
                return z
                
        except Exception:
            # フォールバック値
            if confidence_level == 0.95:
                return 1.645
            elif confidence_level == 0.99:
                return 2.326
            else:
                return 1.96  # 95%のデフォルト
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """歪度計算"""
        try:
            if len(values) < 3:
                return 0.0
            
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            if std_val == 0:
                return 0.0
            
            n = len(values)
            skew = sum((x - mean_val) ** 3 for x in values) / (n * std_val ** 3)
            
            return skew
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """尖度計算"""
        try:
            if len(values) < 4:
                return 0.0
            
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            if std_val == 0:
                return 0.0
            
            n = len(values)
            kurt = sum((x - mean_val) ** 4 for x in values) / (n * std_val ** 4) - 3
            
            return kurt
            
        except Exception:
            return 0.0
    
    def _calculate_autocorrelation(self, values: List[float], lag: int = 1) -> float:
        """自己相関計算"""
        try:
            if len(values) <= lag:
                return 0.0
            
            y1 = values[:-lag]
            y2 = values[lag:]
            
            if len(y1) != len(y2):
                return 0.0
            
            # ピアソン相関係数
            mean_y1 = statistics.mean(y1)
            mean_y2 = statistics.mean(y2)
            
            numerator = sum((x1 - mean_y1) * (x2 - mean_y2) for x1, x2 in zip(y1, y2))
            
            sum_sq_y1 = sum((x1 - mean_y1) ** 2 for x1 in y1)
            sum_sq_y2 = sum((x2 - mean_y2) ** 2 for x2 in y2)
            
            denominator = math.sqrt(sum_sq_y1 * sum_sq_y2)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return correlation
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_persistence(self, volatilities: List[float]) -> float:
        """ボラティリティ永続性計算"""
        try:
            if len(volatilities) < 10:
                return 0.0
            
            # 1期ラグの自己相関
            autocorr = self._calculate_autocorrelation(volatilities, lag=1)
            
            # 永続性の指標として自己相関を使用
            return abs(autocorr)
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_statistics(self, volatilities: List[float]) -> Dict[str, float]:
        """ボラティリティ統計計算"""
        try:
            if not volatilities:
                return {}
            
            return {
                'mean': statistics.mean(volatilities),
                'median': statistics.median(volatilities),
                'std': statistics.stdev(volatilities) if len(volatilities) > 1 else 0.0,
                'min': min(volatilities),
                'max': max(volatilities),
                'range': max(volatilities) - min(volatilities),
                'coefficient_of_variation': statistics.stdev(volatilities) / statistics.mean(volatilities) if len(volatilities) > 1 and statistics.mean(volatilities) != 0 else 0.0
            }
            
        except Exception:
            return {}
    
    def _calculate_volatility_confidence(self, data_points: int, current_vol: float) -> float:
        """ボラティリティ分析の信頼度計算"""
        try:
            # データ点数による信頼度
            data_confidence = min(data_points / 100, 1.0)  # 100点で最大
            
            # ボラティリティの妥当性チェック
            vol_confidence = 1.0 if 0.01 <= current_vol <= 2.0 else 0.5
            
            return data_confidence * vol_confidence
            
        except Exception:
            return 0.5