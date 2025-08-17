"""
Seasonal Analysis Module

季節性分析 - 周期性検出、季節調整、フーリエ分析
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import statistics
from collections import defaultdict

# 依存ライブラリ（条件付きインポート）
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class SeasonalComponent:
    """季節成分"""
    period: int
    amplitude: float
    phase: float
    strength: float
    confidence: float


class SeasonalAnalyzer:
    """
    季節性分析クラス
    
    Features:
    - 季節性検出
    - 季節調整
    - 周期成分分解
    - フーリエ分析（簡易版）
    - 月次・曜日効果分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        季節性アナライザー初期化
        
        Args:
            config: 設定パラメータ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.common_periods = self.config.get('common_periods', [7, 30, 90, 252, 365])  # 週、月、四半期、年（営業日）、年（暦日）
        self.min_cycles = self.config.get('min_cycles', 2)  # 最小サイクル数
        self.significance_threshold = self.config.get('significance_threshold', 0.1)
        
        self.logger.debug("SeasonalAnalyzer initialized")
    
    def analyze_seasonality(self, data) -> 'AnalysisResult':
        """包括的季節性分析"""
        try:
            from .analyzer import AnalysisResult
            
            # 季節性検出
            seasonal_components = self.detect_seasonal_patterns(data)
            
            # 月次効果分析
            monthly_effects = self.analyze_monthly_effects(data)
            
            # 曜日効果分析
            weekday_effects = self.analyze_weekday_effects(data)
            
            # 季節調整
            seasonally_adjusted = self.seasonal_adjustment(data, seasonal_components)
            
            # フーリエ分析（簡易版）
            fourier_analysis = self.fourier_analysis(data)
            
            # 周期性強度
            seasonality_strength = self.calculate_seasonality_strength(seasonal_components)
            
            result_data = {
                'seasonal_components': seasonal_components,
                'seasonality_strength': seasonality_strength,
                'monthly_effects': monthly_effects,
                'weekday_effects': weekday_effects,
                'seasonally_adjusted_data': seasonally_adjusted,
                'fourier_analysis': fourier_analysis,
                'dominant_periods': self._extract_dominant_periods(seasonal_components),
                'seasonal_summary': self._create_seasonal_summary(seasonal_components, monthly_effects, weekday_effects)
            }
            
            # 信頼度計算
            confidence = self._calculate_seasonal_confidence(seasonal_components, len(data.values))
            
            return AnalysisResult(
                analysis_type='seasonality',
                result_data=result_data,
                confidence=confidence,
                metadata={
                    'data_points': len(data.values),
                    'analysis_periods': self.common_periods,
                    'frequency_estimate': self._estimate_data_frequency(data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Seasonality analysis failed: {str(e)}")
            raise
    
    def detect_seasonal_patterns(self, data) -> List[Dict[str, Any]]:
        """季節パターン検出"""
        try:
            seasonal_components = []
            values = data.values
            
            for period in self.common_periods:
                # データ長が周期の2倍以上必要
                if len(values) < period * self.min_cycles:
                    continue
                
                # 自己相関による周期検出
                seasonal_autocorr = self._calculate_seasonal_autocorrelation(values, period)
                
                # 周期的パターンの強度計算
                pattern_strength = self._calculate_pattern_strength(values, period)
                
                # 振幅と位相の推定
                amplitude, phase = self._estimate_amplitude_phase(values, period)
                
                # 統計的有意性
                significance = self._test_seasonal_significance(values, period)
                
                if seasonal_autocorr > self.significance_threshold and significance:
                    seasonal_components.append({
                        'period': period,
                        'autocorrelation': seasonal_autocorr,
                        'pattern_strength': pattern_strength,
                        'amplitude': amplitude,
                        'phase': phase,
                        'significance': significance,
                        'period_name': self._get_period_name(period)
                    })
            
            # 強度順にソート
            seasonal_components.sort(key=lambda x: x['pattern_strength'], reverse=True)
            
            return seasonal_components
            
        except Exception as e:
            self.logger.error(f"Seasonal pattern detection failed: {str(e)}")
            return []
    
    def analyze_monthly_effects(self, data) -> Dict[str, Any]:
        """月次効果分析"""
        try:
            if len(data.timestamps) < 30:
                return {'monthly_effects': False, 'reason': 'insufficient_data'}
            
            # 月別データ集計
            monthly_data = defaultdict(list)
            
            for timestamp, value in zip(data.timestamps, data.values):
                month = timestamp.month
                monthly_data[month].append(value)
            
            # 月別統計
            monthly_stats = {}
            for month in range(1, 13):
                if month in monthly_data and len(monthly_data[month]) > 0:
                    monthly_stats[month] = {
                        'mean': statistics.mean(monthly_data[month]),
                        'median': statistics.median(monthly_data[month]),
                        'std': statistics.stdev(monthly_data[month]) if len(monthly_data[month]) > 1 else 0.0,
                        'count': len(monthly_data[month])
                    }
            
            # 全体平均
            overall_mean = statistics.mean(data.values)
            
            # 月次効果（偏差）
            monthly_effects = {}
            for month, stats in monthly_stats.items():
                effect = (stats['mean'] - overall_mean) / overall_mean * 100  # パーセンテージ
                monthly_effects[month] = {
                    'effect_pct': effect,
                    'absolute_effect': stats['mean'] - overall_mean,
                    'sample_size': stats['count'],
                    'month_name': self._get_month_name(month)
                }
            
            # 月次効果の統計的検定
            monthly_significance = self._test_monthly_significance(monthly_data, overall_mean)
            
            # 最強・最弱月
            if monthly_effects:
                best_month = max(monthly_effects.items(), key=lambda x: x[1]['effect_pct'])
                worst_month = min(monthly_effects.items(), key=lambda x: x[1]['effect_pct'])
            else:
                best_month = worst_month = None
            
            return {
                'monthly_effects': monthly_effects,
                'monthly_significance': monthly_significance,
                'best_month': best_month,
                'worst_month': worst_month,
                'seasonal_variation': max(monthly_effects.values(), key=lambda x: abs(x['effect_pct']))['effect_pct'] if monthly_effects else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Monthly effects analysis failed: {str(e)}")
            return {'monthly_effects': False, 'error': str(e)}
    
    def analyze_weekday_effects(self, data) -> Dict[str, Any]:
        """曜日効果分析"""
        try:
            if len(data.timestamps) < 14:
                return {'weekday_effects': False, 'reason': 'insufficient_data'}
            
            # 曜日別データ集計
            weekday_data = defaultdict(list)
            
            for timestamp, value in zip(data.timestamps, data.values):
                weekday = timestamp.weekday()  # 0=月曜日, 6=日曜日
                weekday_data[weekday].append(value)
            
            # 曜日別統計
            weekday_stats = {}
            for weekday in range(7):
                if weekday in weekday_data and len(weekday_data[weekday]) > 0:
                    weekday_stats[weekday] = {
                        'mean': statistics.mean(weekday_data[weekday]),
                        'median': statistics.median(weekday_data[weekday]),
                        'std': statistics.stdev(weekday_data[weekday]) if len(weekday_data[weekday]) > 1 else 0.0,
                        'count': len(weekday_data[weekday])
                    }
            
            # 全体平均
            overall_mean = statistics.mean(data.values)
            
            # 曜日効果（偏差）
            weekday_effects = {}
            for weekday, stats in weekday_stats.items():
                effect = (stats['mean'] - overall_mean) / overall_mean * 100
                weekday_effects[weekday] = {
                    'effect_pct': effect,
                    'absolute_effect': stats['mean'] - overall_mean,
                    'sample_size': stats['count'],
                    'weekday_name': self._get_weekday_name(weekday)
                }
            
            # 曜日効果の統計的検定
            weekday_significance = self._test_weekday_significance(weekday_data, overall_mean)
            
            # 月曜日効果、金曜日効果の特別チェック
            monday_effect = weekday_effects.get(0, {}).get('effect_pct', 0.0)
            friday_effect = weekday_effects.get(4, {}).get('effect_pct', 0.0)
            
            return {
                'weekday_effects': weekday_effects,
                'weekday_significance': weekday_significance,
                'monday_effect': monday_effect,
                'friday_effect': friday_effect,
                'weekend_vs_weekday': self._analyze_weekend_effect(weekday_data)
            }
            
        except Exception as e:
            self.logger.error(f"Weekday effects analysis failed: {str(e)}")
            return {'weekday_effects': False, 'error': str(e)}
    
    def seasonal_adjustment(self, data, seasonal_components: List[Dict[str, Any]]):
        """季節調整"""
        try:
            from .analyzer import TimeSeriesData
            
            values = data.values.copy()
            
            if not seasonal_components:
                # 季節成分なしの場合、元データを返す
                return TimeSeriesData(
                    timestamps=data.timestamps,
                    values=values,
                    name=f"{data.name}_seasonally_adjusted",
                    metadata={'adjustment_applied': False}
                )
            
            # 主要な季節成分を除去
            for component in seasonal_components[:3]:  # 上位3成分まで
                period = component['period']
                amplitude = component['amplitude']
                phase = component['phase']
                
                # 季節成分の生成と除去
                for i in range(len(values)):
                    seasonal_value = amplitude * math.sin(2 * math.pi * i / period + phase)
                    values[i] -= seasonal_value
            
            return TimeSeriesData(
                timestamps=data.timestamps,
                values=values,
                name=f"{data.name}_seasonally_adjusted",
                metadata={
                    'adjustment_applied': True,
                    'removed_components': len(seasonal_components[:3]),
                    'components_removed': [c['period'] for c in seasonal_components[:3]]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Seasonal adjustment failed: {str(e)}")
            # エラー時は元データを返す
            return data
    
    def fourier_analysis(self, data, max_frequencies: int = 10) -> Dict[str, Any]:
        """フーリエ分析（簡易版）"""
        try:
            values = data.values
            n = len(values)
            
            if n < 8:
                return {'fourier_analysis': False, 'reason': 'insufficient_data'}
            
            # 離散フーリエ変換（DFT）の簡易実装
            frequencies = []
            amplitudes = []
            phases = []
            
            # ナイキスト周波数まで
            max_freq = min(max_frequencies, n // 2)
            
            for k in range(1, max_freq + 1):
                # DFT計算
                real_part = sum(values[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
                imag_part = sum(values[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
                
                amplitude = math.sqrt(real_part ** 2 + imag_part ** 2) / n
                phase = math.atan2(imag_part, real_part)
                frequency = k / n
                
                frequencies.append(frequency)
                amplitudes.append(amplitude)
                phases.append(phase)
            
            # 周波数成分を振幅順にソート
            freq_components = list(zip(frequencies, amplitudes, phases))
            freq_components.sort(key=lambda x: x[1], reverse=True)
            
            # 主要な周波数成分
            dominant_frequencies = freq_components[:5]
            
            # 対応する周期
            dominant_periods = []
            for freq, amp, phase in dominant_frequencies:
                if freq > 0:
                    period = 1 / freq
                    dominant_periods.append({
                        'period': period,
                        'frequency': freq,
                        'amplitude': amp,
                        'phase': phase,
                        'period_interpretation': self._interpret_period(period, len(values))
                    })
            
            # スペクトル密度
            power_spectrum = [amp ** 2 for amp in amplitudes]
            total_power = sum(power_spectrum)
            
            return {
                'dominant_frequencies': dominant_frequencies,
                'dominant_periods': dominant_periods,
                'power_spectrum': power_spectrum,
                'total_power': total_power,
                'spectral_centroid': self._calculate_spectral_centroid(frequencies, power_spectrum),
                'bandwidth': self._calculate_bandwidth(frequencies, power_spectrum)
            }
            
        except Exception as e:
            self.logger.error(f"Fourier analysis failed: {str(e)}")
            return {'fourier_analysis': False, 'error': str(e)}
    
    def calculate_seasonality_strength(self, seasonal_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """季節性強度計算"""
        try:
            if not seasonal_components:
                return {
                    'overall_strength': 0.0,
                    'strength_level': 'none',
                    'dominant_period': None
                }
            
            # 各成分の寄与度を重み付け平均
            weights = [comp['pattern_strength'] for comp in seasonal_components]
            autocorrs = [comp['autocorrelation'] for comp in seasonal_components]
            
            if not weights:
                overall_strength = 0.0
            else:
                overall_strength = sum(w * a for w, a in zip(weights, autocorrs)) / sum(weights)
            
            # 強度レベル分類
            if overall_strength < 0.1:
                strength_level = 'none'
            elif overall_strength < 0.3:
                strength_level = 'weak'
            elif overall_strength < 0.6:
                strength_level = 'moderate'
            else:
                strength_level = 'strong'
            
            # 主要周期
            dominant_period = seasonal_components[0]['period'] if seasonal_components else None
            
            return {
                'overall_strength': overall_strength,
                'strength_level': strength_level,
                'dominant_period': dominant_period,
                'number_of_components': len(seasonal_components),
                'component_contributions': [
                    {
                        'period': comp['period'],
                        'contribution': comp['pattern_strength'],
                        'period_name': comp.get('period_name', f"Period_{comp['period']}")
                    }
                    for comp in seasonal_components
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality strength calculation failed: {str(e)}")
            return {'overall_strength': 0.0, 'strength_level': 'unknown'}
    
    def _calculate_seasonal_autocorrelation(self, values: List[float], period: int) -> float:
        """季節的自己相関計算"""
        try:
            if len(values) < period * 2:
                return 0.0
            
            # ラグがperiodの自己相関
            y1 = values[:-period]
            y2 = values[period:]
            
            if len(y1) != len(y2) or len(y1) < 2:
                return 0.0
            
            # ピアソン相関
            mean1 = statistics.mean(y1)
            mean2 = statistics.mean(y2)
            
            numerator = sum((x1 - mean1) * (x2 - mean2) for x1, x2 in zip(y1, y2))
            
            sum_sq1 = sum((x1 - mean1) ** 2 for x1 in y1)
            sum_sq2 = sum((x2 - mean2) ** 2 for x2 in y2)
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_strength(self, values: List[float], period: int) -> float:
        """パターン強度計算"""
        try:
            if len(values) < period * 2:
                return 0.0
            
            # 周期ごとの平均パターンを計算
            cycles = len(values) // period
            pattern_sum = [0.0] * period
            
            for i in range(cycles):
                for j in range(period):
                    idx = i * period + j
                    if idx < len(values):
                        pattern_sum[j] += values[idx]
            
            # 平均パターン
            avg_pattern = [s / cycles for s in pattern_sum]
            
            # パターンの分散
            pattern_variance = statistics.variance(avg_pattern) if len(avg_pattern) > 1 else 0.0
            
            # 全体の分散
            total_variance = statistics.variance(values) if len(values) > 1 else 0.0
            
            # 強度（パターン分散 / 全体分散）
            if total_variance > 0:
                strength = pattern_variance / total_variance
                return min(strength, 1.0)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _estimate_amplitude_phase(self, values: List[float], period: int) -> Tuple[float, float]:
        """振幅と位相の推定"""
        try:
            if len(values) < period:
                return 0.0, 0.0
            
            n = len(values)
            
            # フーリエ係数計算
            a = sum(values[i] * math.cos(2 * math.pi * i / period) for i in range(n)) * 2 / n
            b = sum(values[i] * math.sin(2 * math.pi * i / period) for i in range(n)) * 2 / n
            
            amplitude = math.sqrt(a ** 2 + b ** 2)
            phase = math.atan2(b, a)
            
            return amplitude, phase
            
        except Exception:
            return 0.0, 0.0
    
    def _test_seasonal_significance(self, values: List[float], period: int) -> bool:
        """季節性の統計的有意性検定（簡易版）"""
        try:
            seasonal_autocorr = self._calculate_seasonal_autocorrelation(values, period)
            
            # 有意性の閾値（95%信頼水準での近似）
            n_effective = len(values) - period
            threshold = 1.96 / math.sqrt(n_effective) if n_effective > 0 else 0.5
            
            return abs(seasonal_autocorr) > threshold
            
        except Exception:
            return False
    
    def _test_monthly_significance(self, monthly_data: Dict[int, List[float]], overall_mean: float) -> Dict[str, Any]:
        """月次効果の統計的有意性検定"""
        try:
            # 月間の平均値
            monthly_means = []
            monthly_counts = []
            
            for month in range(1, 13):
                if month in monthly_data and len(monthly_data[month]) > 0:
                    monthly_means.append(statistics.mean(monthly_data[month]))
                    monthly_counts.append(len(monthly_data[month]))
            
            if len(monthly_means) < 3:
                return {'significant': False, 'reason': 'insufficient_months'}
            
            # 分散分析の簡易版
            grand_mean = statistics.mean(monthly_means)
            between_variance = sum(count * (mean - grand_mean) ** 2 for mean, count in zip(monthly_means, monthly_counts))
            
            # 簡易的な有意性判定
            max_deviation = max(abs(mean - grand_mean) for mean in monthly_means)
            threshold = statistics.stdev(monthly_means) if len(monthly_means) > 1 else 0.0
            
            significant = max_deviation > threshold * 1.5 if threshold > 0 else False
            
            return {
                'significant': significant,
                'max_deviation': max_deviation,
                'threshold': threshold,
                'between_variance': between_variance
            }
            
        except Exception:
            return {'significant': False, 'error': 'calculation_failed'}
    
    def _test_weekday_significance(self, weekday_data: Dict[int, List[float]], overall_mean: float) -> Dict[str, Any]:
        """曜日効果の統計的有意性検定"""
        try:
            weekday_means = []
            for weekday in range(7):
                if weekday in weekday_data and len(weekday_data[weekday]) > 0:
                    weekday_means.append(statistics.mean(weekday_data[weekday]))
            
            if len(weekday_means) < 3:
                return {'significant': False, 'reason': 'insufficient_weekdays'}
            
            # 曜日間の分散
            weekday_variance = statistics.variance(weekday_means) if len(weekday_means) > 1 else 0.0
            
            # 簡易的な有意性判定
            max_deviation = max(abs(mean - overall_mean) for mean in weekday_means)
            threshold = math.sqrt(weekday_variance) * 1.5
            
            significant = max_deviation > threshold if threshold > 0 else False
            
            return {
                'significant': significant,
                'weekday_variance': weekday_variance,
                'max_deviation': max_deviation,
                'threshold': threshold
            }
            
        except Exception:
            return {'significant': False, 'error': 'calculation_failed'}
    
    def _analyze_weekend_effect(self, weekday_data: Dict[int, List[float]]) -> Dict[str, Any]:
        """週末効果分析"""
        try:
            # 平日（月-金）vs 週末（土日）
            weekday_values = []
            weekend_values = []
            
            for day, values in weekday_data.items():
                if day < 5:  # 月-金
                    weekday_values.extend(values)
                else:  # 土日
                    weekend_values.extend(values)
            
            if not weekday_values or not weekend_values:
                return {'weekend_effect': False, 'reason': 'insufficient_data'}
            
            weekday_mean = statistics.mean(weekday_values)
            weekend_mean = statistics.mean(weekend_values)
            
            effect_pct = (weekend_mean - weekday_mean) / weekday_mean * 100 if weekday_mean != 0 else 0.0
            
            return {
                'weekend_effect': True,
                'weekday_mean': weekday_mean,
                'weekend_mean': weekend_mean,
                'effect_pct': effect_pct,
                'weekday_count': len(weekday_values),
                'weekend_count': len(weekend_values)
            }
            
        except Exception:
            return {'weekend_effect': False, 'error': 'calculation_failed'}
    
    def _estimate_data_frequency(self, data) -> str:
        """データ頻度推定"""
        try:
            if len(data.timestamps) < 2:
                return 'unknown'
            
            # 隣接する時点の差分
            deltas = []
            for i in range(1, min(10, len(data.timestamps))):
                delta = data.timestamps[i] - data.timestamps[i-1]
                deltas.append(delta.total_seconds())
            
            avg_delta = statistics.mean(deltas)
            
            if avg_delta <= 60:
                return 'minute'
            elif avg_delta <= 3600:
                return 'hourly'
            elif avg_delta <= 86400:
                return 'daily'
            elif avg_delta <= 604800:
                return 'weekly'
            else:
                return 'monthly'
                
        except Exception:
            return 'unknown'
    
    def _get_period_name(self, period: int) -> str:
        """周期名取得"""
        period_names = {
            7: 'Weekly',
            30: 'Monthly',
            90: 'Quarterly',
            252: 'Yearly (Trading Days)',
            365: 'Yearly (Calendar Days)'
        }
        return period_names.get(period, f'Period_{period}')
    
    def _get_month_name(self, month: int) -> str:
        """月名取得"""
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        return month_names.get(month, f'Month_{month}')
    
    def _get_weekday_name(self, weekday: int) -> str:
        """曜日名取得"""
        weekday_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        return weekday_names.get(weekday, f'Weekday_{weekday}')
    
    def _interpret_period(self, period: float, data_length: int) -> str:
        """周期の解釈"""
        if period < 2:
            return 'Very Short Term'
        elif period < 7:
            return 'Short Term'
        elif period < 30:
            return 'Medium Term'
        elif period < 90:
            return 'Seasonal'
        elif period < 365:
            return 'Annual'
        else:
            return 'Long Term'
    
    def _calculate_spectral_centroid(self, frequencies: List[float], power_spectrum: List[float]) -> float:
        """スペクトル重心計算"""
        try:
            if not frequencies or not power_spectrum or sum(power_spectrum) == 0:
                return 0.0
            
            weighted_sum = sum(f * p for f, p in zip(frequencies, power_spectrum))
            total_power = sum(power_spectrum)
            
            return weighted_sum / total_power
            
        except Exception:
            return 0.0
    
    def _calculate_bandwidth(self, frequencies: List[float], power_spectrum: List[float]) -> float:
        """スペクトル帯域幅計算"""
        try:
            if not frequencies or not power_spectrum:
                return 0.0
            
            centroid = self._calculate_spectral_centroid(frequencies, power_spectrum)
            total_power = sum(power_spectrum)
            
            if total_power == 0:
                return 0.0
            
            variance = sum(((f - centroid) ** 2) * p for f, p in zip(frequencies, power_spectrum)) / total_power
            bandwidth = math.sqrt(variance)
            
            return bandwidth
            
        except Exception:
            return 0.0
    
    def _extract_dominant_periods(self, seasonal_components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """主要周期の抽出"""
        try:
            if not seasonal_components:
                return []
            
            # 上位3つの主要周期
            dominant = seasonal_components[:3]
            
            return [
                {
                    'period': comp['period'],
                    'period_name': comp.get('period_name', f"Period_{comp['period']}"),
                    'strength': comp['pattern_strength'],
                    'autocorrelation': comp['autocorrelation']
                }
                for comp in dominant
            ]
            
        except Exception:
            return []
    
    def _create_seasonal_summary(self, seasonal_components: List[Dict[str, Any]], 
                                monthly_effects: Dict[str, Any], 
                                weekday_effects: Dict[str, Any]) -> Dict[str, Any]:
        """季節性サマリー作成"""
        try:
            summary = {
                'has_seasonality': len(seasonal_components) > 0,
                'seasonal_strength': 'none'
            }
            
            if seasonal_components:
                max_strength = max(comp['pattern_strength'] for comp in seasonal_components)
                
                if max_strength > 0.6:
                    summary['seasonal_strength'] = 'strong'
                elif max_strength > 0.3:
                    summary['seasonal_strength'] = 'moderate'
                else:
                    summary['seasonal_strength'] = 'weak'
                
                summary['primary_period'] = seasonal_components[0]['period']
                summary['primary_period_name'] = seasonal_components[0].get('period_name')
            
            # 月次・曜日効果
            summary['has_monthly_effects'] = monthly_effects.get('monthly_significance', {}).get('significant', False)
            summary['has_weekday_effects'] = weekday_effects.get('weekday_significance', {}).get('significant', False)
            
            # 特別な効果
            if weekday_effects.get('monday_effect', 0) != 0:
                summary['monday_effect'] = True
            
            if monthly_effects.get('best_month') and monthly_effects.get('worst_month'):
                summary['seasonal_months'] = {
                    'best': monthly_effects['best_month'][1]['month_name'],
                    'worst': monthly_effects['worst_month'][1]['month_name']
                }
            
            return summary
            
        except Exception:
            return {'has_seasonality': False, 'seasonal_strength': 'unknown'}
    
    def _calculate_seasonal_confidence(self, seasonal_components: List[Dict[str, Any]], data_points: int) -> float:
        """季節性分析の信頼度計算"""
        try:
            # データ点数による信頼度
            data_confidence = min(data_points / 100, 1.0)
            
            # 検出された季節成分の品質
            if not seasonal_components:
                component_confidence = 0.3
            else:
                # 最強成分の品質
                max_strength = max(comp['pattern_strength'] for comp in seasonal_components)
                max_autocorr = max(comp['autocorrelation'] for comp in seasonal_components)
                component_confidence = (max_strength + max_autocorr) / 2
            
            # サイクル数による信頼度
            if seasonal_components:
                min_period = min(comp['period'] for comp in seasonal_components)
                cycles = data_points / min_period
                cycle_confidence = min(cycles / 3, 1.0)  # 3サイクル以上で最大
            else:
                cycle_confidence = 0.5
            
            # 総合信頼度
            total_confidence = (data_confidence + component_confidence + cycle_confidence) / 3
            
            return max(0.0, min(1.0, total_confidence))
            
        except Exception:
            return 0.5