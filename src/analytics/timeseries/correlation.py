"""
Correlation Analysis Module

相関分析 - ピアソン、スピアマン、動的相関、クロス相関分析
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
class CorrelationResult:
    """相関分析結果"""
    correlation_type: str
    correlation_value: float
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    significance_level: float
    metadata: Dict[str, Any]


class CorrelationAnalyzer:
    """
    相関分析クラス
    
    Features:
    - ピアソン相関係数
    - スピアマン順位相関
    - 自己相関分析
    - クロス相関分析
    - 動的相関（ローリング相関）
    - 相関行列分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        相関アナライザー初期化
        
        Args:
            config: 設定パラメータ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.default_window = self.config.get('correlation_window', 20)
        self.max_lags = self.config.get('max_lags', 20)
        self.significance_level = self.config.get('significance_level', 0.05)
        
        self.logger.debug("CorrelationAnalyzer initialized")
    
    def analyze_correlation(self, data1, data2, correlation_types: Optional[List[str]] = None) -> Dict[str, 'AnalysisResult']:
        """包括的相関分析"""
        try:
            from .analyzer import AnalysisResult
            
            if correlation_types is None:
                correlation_types = ['pearson', 'spearman', 'rolling', 'cross_correlation']
            
            results = {}
            
            # データの同期（共通期間の抽出）
            aligned_data1, aligned_data2 = self._align_timeseries(data1, data2)
            
            if len(aligned_data1.values) < 2:
                raise ValueError("Insufficient aligned data for correlation analysis")
            
            # ピアソン相関
            if 'pearson' in correlation_types:
                pearson_result = self.calculate_pearson_correlation(aligned_data1, aligned_data2)
                results['pearson'] = AnalysisResult(
                    analysis_type='pearson_correlation',
                    result_data=pearson_result,
                    confidence=self._calculate_correlation_confidence(pearson_result, len(aligned_data1.values))
                )
            
            # スピアマン相関
            if 'spearman' in correlation_types:
                spearman_result = self.calculate_spearman_correlation(aligned_data1, aligned_data2)
                results['spearman'] = AnalysisResult(
                    analysis_type='spearman_correlation',
                    result_data=spearman_result,
                    confidence=self._calculate_correlation_confidence(spearman_result, len(aligned_data1.values))
                )
            
            # ローリング相関
            if 'rolling' in correlation_types:
                rolling_result = self.calculate_rolling_correlation(aligned_data1, aligned_data2)
                results['rolling'] = AnalysisResult(
                    analysis_type='rolling_correlation',
                    result_data=rolling_result,
                    confidence=0.8  # ローリング相関の信頼度
                )
            
            # クロス相関
            if 'cross_correlation' in correlation_types:
                cross_corr_result = self.calculate_cross_correlation(aligned_data1, aligned_data2)
                results['cross_correlation'] = AnalysisResult(
                    analysis_type='cross_correlation',
                    result_data=cross_corr_result,
                    confidence=0.75
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
            raise
    
    def analyze_autocorrelation(self, data, max_lags: Optional[int] = None) -> 'AnalysisResult':
        """自己相関分析"""
        try:
            from .analyzer import AnalysisResult
            
            if max_lags is None:
                max_lags = min(self.max_lags, len(data.values) // 4)
            
            autocorrelations = []
            significant_lags = []
            
            for lag in range(1, max_lags + 1):
                autocorr = self._calculate_autocorrelation(data.values, lag)
                autocorrelations.append({
                    'lag': lag,
                    'autocorrelation': autocorr,
                    'significant': abs(autocorr) > self._get_autocorr_threshold(len(data.values))
                })
                
                if abs(autocorr) > self._get_autocorr_threshold(len(data.values)):
                    significant_lags.append(lag)
            
            # リュング=ボックス統計量（簡易版）
            ljung_box_stat = self._calculate_ljung_box_statistic(data.values, max_lags)
            
            result_data = {
                'autocorrelations': autocorrelations,
                'significant_lags': significant_lags,
                'ljung_box_statistic': ljung_box_stat,
                'serial_correlation_detected': len(significant_lags) > 0,
                'max_autocorr_lag': max(autocorrelations, key=lambda x: abs(x['autocorrelation']))['lag'] if autocorrelations else 0,
                'max_autocorr_value': max(autocorrelations, key=lambda x: abs(x['autocorrelation']))['autocorrelation'] if autocorrelations else 0.0
            }
            
            confidence = 0.9 if len(data.values) > 50 else 0.7
            
            return AnalysisResult(
                analysis_type='autocorrelation',
                result_data=result_data,
                confidence=confidence,
                metadata={'max_lags': max_lags, 'data_points': len(data.values)}
            )
            
        except Exception as e:
            self.logger.error(f"Autocorrelation analysis failed: {str(e)}")
            raise
    
    def calculate_pearson_correlation(self, data1, data2) -> Dict[str, Any]:
        """ピアソン相関係数計算"""
        try:
            values1 = data1.values
            values2 = data2.values
            
            if len(values1) != len(values2):
                raise ValueError("Data series must have same length")
            
            if len(values1) < 2:
                return {'correlation': 0.0, 'error': 'insufficient_data'}
            
            # ピアソン相関係数
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            
            numerator = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(values1, values2))
            
            sum_sq1 = sum((x1 - mean1) ** 2 for x1 in values1)
            sum_sq2 = sum((x2 - mean2) ** 2 for x2 in values2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
            
            # t統計量とp値（近似）
            n = len(values1)
            if n > 2 and abs(correlation) < 1:
                t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2))
                p_value = self._calculate_t_test_p_value(t_stat, n - 2)
            else:
                t_stat = 0.0
                p_value = 1.0
            
            # 信頼区間（Fisher変換）
            confidence_interval = self._calculate_correlation_confidence_interval(correlation, n)
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                't_statistic': t_stat,
                'confidence_interval': confidence_interval,
                'significant': p_value < self.significance_level,
                'sample_size': n,
                'correlation_strength': self._classify_correlation_strength(abs(correlation))
            }
            
        except Exception as e:
            self.logger.error(f"Pearson correlation calculation failed: {str(e)}")
            return {'correlation': 0.0, 'error': str(e)}
    
    def calculate_spearman_correlation(self, data1, data2) -> Dict[str, Any]:
        """スピアマン順位相関係数計算"""
        try:
            values1 = data1.values
            values2 = data2.values
            
            if len(values1) != len(values2):
                raise ValueError("Data series must have same length")
            
            if len(values1) < 2:
                return {'correlation': 0.0, 'error': 'insufficient_data'}
            
            # 順位付け
            ranks1 = self._rank_data(values1)
            ranks2 = self._rank_data(values2)
            
            # ピアソン相関を順位データに適用
            mean_rank1 = statistics.mean(ranks1)
            mean_rank2 = statistics.mean(ranks2)
            
            numerator = sum((r1 - mean_rank1) * (r2 - mean_rank2) for r1, r2 in zip(ranks1, ranks2))
            
            sum_sq1 = sum((r1 - mean_rank1) ** 2 for r1 in ranks1)
            sum_sq2 = sum((r2 - mean_rank2) ** 2 for r2 in ranks2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
            
            # 統計的検定
            n = len(values1)
            if n > 2:
                t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2))
                p_value = self._calculate_t_test_p_value(t_stat, n - 2)
            else:
                t_stat = 0.0
                p_value = 1.0
            
            return {
                'correlation': correlation,
                'p_value': p_value,
                't_statistic': t_stat,
                'significant': p_value < self.significance_level,
                'sample_size': n,
                'correlation_strength': self._classify_correlation_strength(abs(correlation))
            }
            
        except Exception as e:
            self.logger.error(f"Spearman correlation calculation failed: {str(e)}")
            return {'correlation': 0.0, 'error': str(e)}
    
    def calculate_rolling_correlation(self, data1, data2, window: Optional[int] = None) -> Dict[str, Any]:
        """ローリング相関計算"""
        try:
            from .analyzer import TimeSeriesData
            
            if window is None:
                window = self.default_window
            
            values1 = data1.values
            values2 = data2.values
            timestamps = data1.timestamps
            
            if len(values1) < window:
                return {'error': 'insufficient_data', 'required_window': window, 'available_data': len(values1)}
            
            rolling_correlations = []
            rolling_timestamps = []
            
            for i in range(window - 1, len(values1)):
                window_values1 = values1[i - window + 1:i + 1]
                window_values2 = values2[i - window + 1:i + 1]
                
                # ピアソン相関計算
                try:
                    correlation = self._calculate_pearson_simple(window_values1, window_values2)
                    rolling_correlations.append(correlation)
                    rolling_timestamps.append(timestamps[i])
                except:
                    rolling_correlations.append(0.0)
                    rolling_timestamps.append(timestamps[i])
            
            rolling_corr_data = TimeSeriesData(
                timestamps=rolling_timestamps,
                values=rolling_correlations,
                name=f"rolling_correlation_{window}",
                metadata={'window': window}
            )
            
            # 統計サマリー
            correlation_stats = {
                'mean': statistics.mean(rolling_correlations),
                'std': statistics.stdev(rolling_correlations) if len(rolling_correlations) > 1 else 0.0,
                'min': min(rolling_correlations),
                'max': max(rolling_correlations),
                'current': rolling_correlations[-1] if rolling_correlations else 0.0
            }
            
            # 相関の安定性
            stability = 1.0 - correlation_stats['std']  # 標準偏差が小さいほど安定
            
            return {
                'rolling_correlation_data': rolling_corr_data,
                'correlation_statistics': correlation_stats,
                'correlation_stability': max(0.0, stability),
                'window': window,
                'regime_changes': self._detect_correlation_regime_changes(rolling_correlations)
            }
            
        except Exception as e:
            self.logger.error(f"Rolling correlation calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def calculate_cross_correlation(self, data1, data2, max_lags: Optional[int] = None) -> Dict[str, Any]:
        """クロス相関分析"""
        try:
            if max_lags is None:
                max_lags = min(self.max_lags, len(data1.values) // 4)
            
            values1 = data1.values
            values2 = data2.values
            
            cross_correlations = []
            
            # ラグ 0（同時期）
            lag_0_corr = self._calculate_pearson_simple(values1, values2)
            cross_correlations.append({'lag': 0, 'correlation': lag_0_corr})
            
            # 正のラグ（data2がdata1より先行）
            for lag in range(1, max_lags + 1):
                if lag < len(values2):
                    corr = self._calculate_pearson_simple(values1[:-lag], values2[lag:])
                    cross_correlations.append({'lag': -lag, 'correlation': corr})
            
            # 負のラグ（data1がdata2より先行）
            for lag in range(1, max_lags + 1):
                if lag < len(values1):
                    corr = self._calculate_pearson_simple(values1[lag:], values2[:-lag])
                    cross_correlations.append({'lag': lag, 'correlation': corr})
            
            # 最大相関とそのラグ
            max_corr_entry = max(cross_correlations, key=lambda x: abs(x['correlation']))
            
            # 先行・遅行関係の判定
            if max_corr_entry['lag'] > 0:
                lead_lag_relationship = f"{data1.name} leads {data2.name} by {max_corr_entry['lag']} periods"
            elif max_corr_entry['lag'] < 0:
                lead_lag_relationship = f"{data2.name} leads {data1.name} by {abs(max_corr_entry['lag'])} periods"
            else:
                lead_lag_relationship = "Contemporaneous relationship"
            
            return {
                'cross_correlations': cross_correlations,
                'max_correlation': max_corr_entry['correlation'],
                'optimal_lag': max_corr_entry['lag'],
                'lead_lag_relationship': lead_lag_relationship,
                'synchronous_correlation': lag_0_corr,
                'max_lags_tested': max_lags
            }
            
        except Exception as e:
            self.logger.error(f"Cross correlation calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def calculate_correlation_matrix(self, datasets: List, method: str = 'pearson') -> Dict[str, Any]:
        """複数データセットの相関行列計算"""
        try:
            if len(datasets) < 2:
                raise ValueError("Need at least 2 datasets for correlation matrix")
            
            # 全データを共通期間に整列
            aligned_datasets = self._align_multiple_timeseries(datasets)
            
            n = len(aligned_datasets)
            correlation_matrix = []
            p_value_matrix = []
            
            for i in range(n):
                corr_row = []
                p_val_row = []
                
                for j in range(n):
                    if i == j:
                        corr_row.append(1.0)
                        p_val_row.append(0.0)
                    else:
                        if method == 'pearson':
                            result = self.calculate_pearson_correlation(aligned_datasets[i], aligned_datasets[j])
                        elif method == 'spearman':
                            result = self.calculate_spearman_correlation(aligned_datasets[i], aligned_datasets[j])
                        else:
                            raise ValueError(f"Unknown correlation method: {method}")
                        
                        corr_row.append(result.get('correlation', 0.0))
                        p_val_row.append(result.get('p_value', 1.0))
                
                correlation_matrix.append(corr_row)
                p_value_matrix.append(p_val_row)
            
            # データセット名
            dataset_names = [data.name for data in aligned_datasets]
            
            # 相関行列の特性分析
            matrix_analysis = self._analyze_correlation_matrix(correlation_matrix, dataset_names)
            
            return {
                'correlation_matrix': correlation_matrix,
                'p_value_matrix': p_value_matrix,
                'dataset_names': dataset_names,
                'method': method,
                'matrix_analysis': matrix_analysis,
                'sample_size': len(aligned_datasets[0].values) if aligned_datasets else 0
            }
            
        except Exception as e:
            self.logger.error(f"Correlation matrix calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _align_timeseries(self, data1, data2):
        """2つの時系列データを共通期間に整列"""
        try:
            from .analyzer import TimeSeriesData
            
            # 共通するタイムスタンプを見つける
            timestamps1 = set(data1.timestamps)
            timestamps2 = set(data2.timestamps)
            common_timestamps = sorted(timestamps1.intersection(timestamps2))
            
            if not common_timestamps:
                raise ValueError("No common timestamps found")
            
            # インデックスマッピング
            index_map1 = {ts: i for i, ts in enumerate(data1.timestamps)}
            index_map2 = {ts: i for i, ts in enumerate(data2.timestamps)}
            
            # 整列されたデータ作成
            aligned_values1 = [data1.values[index_map1[ts]] for ts in common_timestamps]
            aligned_values2 = [data2.values[index_map2[ts]] for ts in common_timestamps]
            
            aligned_data1 = TimeSeriesData(
                timestamps=common_timestamps,
                values=aligned_values1,
                name=data1.name
            )
            
            aligned_data2 = TimeSeriesData(
                timestamps=common_timestamps,
                values=aligned_values2,
                name=data2.name
            )
            
            return aligned_data1, aligned_data2
            
        except Exception as e:
            self.logger.error(f"Timeseries alignment failed: {str(e)}")
            raise
    
    def _align_multiple_timeseries(self, datasets):
        """複数の時系列データを共通期間に整列"""
        try:
            from .analyzer import TimeSeriesData
            
            if not datasets:
                return []
            
            # 全データセットに共通するタイムスタンプを見つける
            common_timestamps = set(datasets[0].timestamps)
            for data in datasets[1:]:
                common_timestamps = common_timestamps.intersection(set(data.timestamps))
            
            common_timestamps = sorted(common_timestamps)
            
            if not common_timestamps:
                raise ValueError("No common timestamps found across all datasets")
            
            # 各データセットを整列
            aligned_datasets = []
            for data in datasets:
                index_map = {ts: i for i, ts in enumerate(data.timestamps)}
                aligned_values = [data.values[index_map[ts]] for ts in common_timestamps]
                
                aligned_data = TimeSeriesData(
                    timestamps=common_timestamps,
                    values=aligned_values,
                    name=data.name
                )
                aligned_datasets.append(aligned_data)
            
            return aligned_datasets
            
        except Exception as e:
            self.logger.error(f"Multiple timeseries alignment failed: {str(e)}")
            raise
    
    def _calculate_pearson_simple(self, values1: List[float], values2: List[float]) -> float:
        """シンプルなピアソン相関計算"""
        try:
            if len(values1) != len(values2) or len(values1) < 2:
                return 0.0
            
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            
            numerator = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(values1, values2))
            
            sum_sq1 = sum((x1 - mean1) ** 2 for x1 in values1)
            sum_sq2 = sum((x2 - mean2) ** 2 for x2 in values2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """自己相関計算"""
        try:
            if len(values) <= lag:
                return 0.0
            
            y1 = values[:-lag]
            y2 = values[lag:]
            
            return self._calculate_pearson_simple(y1, y2)
            
        except Exception:
            return 0.0
    
    def _rank_data(self, values: List[float]) -> List[float]:
        """データの順位付け"""
        try:
            # ソートされたインデックス
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            
            # 順位配列
            ranks = [0] * len(values)
            
            for rank, index in enumerate(sorted_indices):
                ranks[index] = rank + 1
            
            return ranks
            
        except Exception:
            return list(range(1, len(values) + 1))
    
    def _get_autocorr_threshold(self, n: int) -> float:
        """自己相関の有意性閾値"""
        try:
            # 95%信頼区間での閾値
            return 1.96 / math.sqrt(n)
        except Exception:
            return 0.2
    
    def _calculate_ljung_box_statistic(self, values: List[float], max_lags: int) -> Dict[str, Any]:
        """リュング=ボックス統計量（簡易版）"""
        try:
            n = len(values)
            autocorrs = []
            
            for lag in range(1, min(max_lags + 1, n // 4)):
                autocorr = self._calculate_autocorrelation(values, lag)
                autocorrs.append(autocorr)
            
            # LB統計量
            lb_stat = n * (n + 2) * sum(autocorr ** 2 / (n - lag - 1) for lag, autocorr in enumerate(autocorrs))
            
            # 自由度
            degrees_of_freedom = len(autocorrs)
            
            # 臨界値（簡易近似）
            critical_value = 2 * degrees_of_freedom  # χ²分布の近似
            
            return {
                'lb_statistic': lb_stat,
                'degrees_of_freedom': degrees_of_freedom,
                'critical_value': critical_value,
                'p_value': 0.05 if lb_stat > critical_value else 0.5,  # 簡易近似
                'serial_correlation_detected': lb_stat > critical_value
            }
            
        except Exception:
            return {'lb_statistic': 0.0, 'serial_correlation_detected': False}
    
    def _calculate_t_test_p_value(self, t_stat: float, df: int) -> float:
        """t検定のp値（簡易近似）"""
        try:
            # 簡易的なp値近似
            abs_t = abs(t_stat)
            
            if df <= 0:
                return 1.0
            
            # 近似式
            if abs_t < 1.0:
                return 0.3
            elif abs_t < 1.5:
                return 0.1
            elif abs_t < 2.0:
                return 0.05
            elif abs_t < 2.5:
                return 0.01
            else:
                return 0.001
                
        except Exception:
            return 0.5
    
    def _calculate_correlation_confidence_interval(self, correlation: float, n: int) -> Tuple[float, float]:
        """相関係数の信頼区間（Fisher変換）"""
        try:
            if n <= 3 or abs(correlation) >= 1:
                return (correlation, correlation)
            
            # Fisher変換
            z = 0.5 * math.log((1 + correlation) / (1 - correlation))
            se = 1 / math.sqrt(n - 3)
            
            # 95%信頼区間
            z_lower = z - 1.96 * se
            z_upper = z + 1.96 * se
            
            # 逆変換
            corr_lower = (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)
            corr_upper = (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1)
            
            return (corr_lower, corr_upper)
            
        except Exception:
            return (correlation - 0.1, correlation + 0.1)
    
    def _classify_correlation_strength(self, abs_correlation: float) -> str:
        """相関の強度分類"""
        if abs_correlation < 0.1:
            return 'negligible'
        elif abs_correlation < 0.3:
            return 'weak'
        elif abs_correlation < 0.5:
            return 'moderate'
        elif abs_correlation < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def _detect_correlation_regime_changes(self, correlations: List[float]) -> List[Dict[str, Any]]:
        """相関レジーム変化検出"""
        try:
            if len(correlations) < 10:
                return []
            
            regime_changes = []
            window = 5
            
            for i in range(window, len(correlations) - window):
                before = statistics.mean(correlations[i-window:i])
                after = statistics.mean(correlations[i:i+window])
                
                change = abs(after - before)
                
                if change > 0.3:  # 30%以上の変化
                    regime_changes.append({
                        'index': i,
                        'correlation_before': before,
                        'correlation_after': after,
                        'change_magnitude': change,
                        'change_type': 'increase' if after > before else 'decrease'
                    })
            
            return regime_changes
            
        except Exception:
            return []
    
    def _analyze_correlation_matrix(self, matrix: List[List[float]], names: List[str]) -> Dict[str, Any]:
        """相関行列の特性分析"""
        try:
            n = len(matrix)
            
            # 上三角行列の相関値を抽出（対角線除く）
            correlations = []
            for i in range(n):
                for j in range(i + 1, n):
                    correlations.append(matrix[i][j])
            
            if not correlations:
                return {}
            
            # 統計サマリー
            stats = {
                'mean_correlation': statistics.mean([abs(c) for c in correlations]),
                'max_correlation': max(correlations),
                'min_correlation': min(correlations),
                'std_correlation': statistics.stdev(correlations) if len(correlations) > 1 else 0.0
            }
            
            # 強い相関ペア
            strong_correlations = []
            for i in range(n):
                for j in range(i + 1, n):
                    corr = matrix[i][j]
                    if abs(corr) > 0.7:
                        strong_correlations.append({
                            'pair': f"{names[i]} - {names[j]}",
                            'correlation': corr,
                            'strength': self._classify_correlation_strength(abs(corr))
                        })
            
            return {
                'statistics': stats,
                'strong_correlations': strong_correlations,
                'highly_correlated_pairs': len(strong_correlations),
                'average_absolute_correlation': statistics.mean([abs(c) for c in correlations])
            }
            
        except Exception:
            return {}
    
    def _calculate_correlation_confidence(self, correlation_result: Dict[str, Any], sample_size: int) -> float:
        """相関分析の信頼度計算"""
        try:
            correlation = correlation_result.get('correlation', 0.0)
            p_value = correlation_result.get('p_value', 1.0)
            
            # サンプルサイズによる信頼度
            size_confidence = min(sample_size / 50, 1.0)
            
            # 統計的有意性による信頼度
            significance_confidence = 1.0 - p_value if p_value < 0.05 else 0.5
            
            # 相関の強度による信頼度
            strength_confidence = abs(correlation)
            
            # 総合信頼度
            total_confidence = (size_confidence + significance_confidence + strength_confidence) / 3
            
            return max(0.0, min(1.0, total_confidence))
            
        except Exception:
            return 0.5