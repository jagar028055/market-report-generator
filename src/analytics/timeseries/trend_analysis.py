"""
Trend Analysis Module

トレンド分析 - 移動平均、指数平滑化、トレンド検出
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
class TrendResult:
    """トレンド分析結果"""
    trend_direction: str  # 'upward', 'downward', 'sideways'
    trend_strength: float  # 0.0-1.0
    slope: float
    r_squared: float
    confidence: float
    metadata: Dict[str, Any]


class TrendAnalyzer:
    """
    トレンド分析クラス
    
    Features:
    - 移動平均（単純、指数、加重）
    - トレンド方向検出
    - トレンド強度計算
    - サポート・レジスタンス検出
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        トレンドアナライザー初期化
        
        Args:
            config: 設定パラメータ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.default_short_window = self.config.get('short_window', 10)
        self.default_long_window = self.config.get('long_window', 20)
        self.trend_threshold = self.config.get('trend_threshold', 0.02)  # 2%
        
        self.logger.debug("TrendAnalyzer initialized")
    
    def analyze_trend(self, data) -> 'AnalysisResult':
        """包括的トレンド分析"""
        try:
            from .analyzer import AnalysisResult
            
            # 移動平均計算
            sma_short = self.calculate_simple_moving_average(data, window=self.default_short_window)
            sma_long = self.calculate_simple_moving_average(data, window=self.default_long_window)
            ema = self.calculate_exponential_moving_average(data, window=self.default_short_window)
            
            # トレンド方向検出
            trend_direction = self.detect_trend_direction(data, window=self.default_long_window)
            
            # トレンド強度計算
            trend_strength = self.calculate_trend_strength(data, window=self.default_long_window)
            
            # 線形回帰
            linear_trend = self.calculate_linear_trend(data)
            
            # 移動平均クロスオーバー
            crossover = self.detect_moving_average_crossover(sma_short, sma_long)
            
            result_data = {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'linear_trend': linear_trend,
                'moving_averages': {
                    'sma_short': {
                        'window': self.default_short_window,
                        'current_value': sma_short.values[-1] if sma_short.values else None,
                        'data': sma_short
                    },
                    'sma_long': {
                        'window': self.default_long_window,
                        'current_value': sma_long.values[-1] if sma_long.values else None,
                        'data': sma_long
                    },
                    'ema': {
                        'window': self.default_short_window,
                        'current_value': ema.values[-1] if ema.values else None,
                        'data': ema
                    }
                },
                'crossover_signals': crossover
            }
            
            # 信頼度計算
            confidence = self._calculate_trend_confidence(linear_trend, trend_strength)
            
            return AnalysisResult(
                analysis_type='trend',
                result_data=result_data,
                confidence=confidence,
                metadata={
                    'data_points': len(data.values),
                    'analysis_period': (data.timestamps[-1] - data.timestamps[0]).days if len(data.timestamps) > 1 else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            raise
    
    def calculate_simple_moving_average(self, data, window: int = 20):
        """単純移動平均計算"""
        try:
            from .analyzer import TimeSeriesData
            
            values = data.values
            timestamps = data.timestamps
            
            if len(values) < window:
                self.logger.warning(f"Data length ({len(values)}) < window size ({window})")
                window = len(values)
            
            sma_values = []
            sma_timestamps = []
            
            for i in range(window - 1, len(values)):
                window_data = values[i - window + 1:i + 1]
                sma_value = statistics.mean(window_data)
                sma_values.append(sma_value)
                sma_timestamps.append(timestamps[i])
            
            return TimeSeriesData(
                timestamps=sma_timestamps,
                values=sma_values,
                name=f"{data.name}_SMA_{window}",
                metadata={'window': window, 'type': 'simple_moving_average'}
            )
            
        except Exception as e:
            self.logger.error(f"Simple moving average calculation failed: {str(e)}")
            raise
    
    def calculate_exponential_moving_average(self, data, window: int = 20):
        """指数移動平均計算"""
        try:
            from .analyzer import TimeSeriesData
            
            values = data.values
            timestamps = data.timestamps
            
            if len(values) == 0:
                raise ValueError("Empty data")
            
            # 平滑化係数
            alpha = 2.0 / (window + 1)
            
            ema_values = []
            ema_timestamps = []
            
            # 初期値は最初の値
            ema = values[0]
            ema_values.append(ema)
            ema_timestamps.append(timestamps[0])
            
            # EMA計算
            for i in range(1, len(values)):
                ema = alpha * values[i] + (1 - alpha) * ema
                ema_values.append(ema)
                ema_timestamps.append(timestamps[i])
            
            return TimeSeriesData(
                timestamps=ema_timestamps,
                values=ema_values,
                name=f"{data.name}_EMA_{window}",
                metadata={'window': window, 'alpha': alpha, 'type': 'exponential_moving_average'}
            )
            
        except Exception as e:
            self.logger.error(f"Exponential moving average calculation failed: {str(e)}")
            raise
    
    def calculate_weighted_moving_average(self, data, window: int = 20):
        """加重移動平均計算"""
        try:
            from .analyzer import TimeSeriesData
            
            values = data.values
            timestamps = data.timestamps
            
            if len(values) < window:
                window = len(values)
            
            # 重み計算（直近のデータほど重い）
            weights = list(range(1, window + 1))
            weight_sum = sum(weights)
            
            wma_values = []
            wma_timestamps = []
            
            for i in range(window - 1, len(values)):
                window_data = values[i - window + 1:i + 1]
                weighted_sum = sum(w * v for w, v in zip(weights, window_data))
                wma_value = weighted_sum / weight_sum
                wma_values.append(wma_value)
                wma_timestamps.append(timestamps[i])
            
            return TimeSeriesData(
                timestamps=wma_timestamps,
                values=wma_values,
                name=f"{data.name}_WMA_{window}",
                metadata={'window': window, 'weights': weights, 'type': 'weighted_moving_average'}
            )
            
        except Exception as e:
            self.logger.error(f"Weighted moving average calculation failed: {str(e)}")
            raise
    
    def detect_trend_direction(self, data, window: int = 20) -> str:
        """トレンド方向検出"""
        try:
            values = data.values
            
            if len(values) < window:
                window = len(values)
            
            if window < 2:
                return 'sideways'
            
            # 最近のウィンドウでの線形回帰
            recent_values = values[-window:]
            x_values = list(range(len(recent_values)))
            
            # 線形回帰の傾き計算
            slope = self._calculate_slope(x_values, recent_values)
            
            # 現在価格と期間開始価格の比較
            price_change_pct = (values[-1] - values[-window]) / values[-window] * 100
            
            # トレンド判定
            if abs(price_change_pct) < self.trend_threshold:
                return 'sideways'
            elif slope > 0 and price_change_pct > self.trend_threshold:
                return 'upward'
            elif slope < 0 and price_change_pct < -self.trend_threshold:
                return 'downward'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"Trend direction detection failed: {str(e)}")
            return 'unknown'
    
    def calculate_trend_strength(self, data, window: int = 20) -> float:
        """トレンド強度計算"""
        try:
            values = data.values
            
            if len(values) < window:
                window = len(values)
            
            if window < 3:
                return 0.0
            
            recent_values = values[-window:]
            x_values = list(range(len(recent_values)))
            
            # 決定係数（R²）計算
            r_squared = self._calculate_r_squared(x_values, recent_values)
            
            # ボラティリティ考慮
            volatility = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            mean_value = statistics.mean(recent_values)
            cv = volatility / mean_value if mean_value != 0 else 0.0  # 変動係数
            
            # トレンド強度（R²をボラティリティで調整）
            strength = r_squared * (1 - min(cv, 1.0))
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation failed: {str(e)}")
            return 0.0
    
    def calculate_linear_trend(self, data) -> Dict[str, float]:
        """線形トレンド計算"""
        try:
            values = data.values
            x_values = list(range(len(values)))
            
            if len(values) < 2:
                return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
            
            # 線形回帰
            slope = self._calculate_slope(x_values, values)
            intercept = self._calculate_intercept(x_values, values, slope)
            r_squared = self._calculate_r_squared(x_values, values)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'trend_line_start': intercept,
                'trend_line_end': intercept + slope * (len(values) - 1)
            }
            
        except Exception as e:
            self.logger.error(f"Linear trend calculation failed: {str(e)}")
            return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
    
    def detect_moving_average_crossover(self, short_ma, long_ma) -> List[Dict[str, Any]]:
        """移動平均クロスオーバー検出"""
        try:
            if not short_ma.values or not long_ma.values:
                return []
            
            # データ長を合わせる
            min_length = min(len(short_ma.values), len(long_ma.values))
            short_values = short_ma.values[-min_length:]
            long_values = long_ma.values[-min_length:]
            timestamps = short_ma.timestamps[-min_length:]
            
            crossovers = []
            
            for i in range(1, len(short_values)):
                prev_short = short_values[i-1]
                curr_short = short_values[i]
                prev_long = long_values[i-1]
                curr_long = long_values[i]
                
                # ゴールデンクロス（短期MAが長期MAを上抜け）
                if prev_short <= prev_long and curr_short > curr_long:
                    crossovers.append({
                        'type': 'golden_cross',
                        'timestamp': timestamps[i],
                        'short_ma_value': curr_short,
                        'long_ma_value': curr_long,
                        'signal': 'bullish'
                    })
                
                # デッドクロス（短期MAが長期MAを下抜け）
                elif prev_short >= prev_long and curr_short < curr_long:
                    crossovers.append({
                        'type': 'dead_cross',
                        'timestamp': timestamps[i],
                        'short_ma_value': curr_short,
                        'long_ma_value': curr_long,
                        'signal': 'bearish'
                    })
            
            return crossovers
            
        except Exception as e:
            self.logger.error(f"Moving average crossover detection failed: {str(e)}")
            return []
    
    def detect_support_resistance(self, data, window: int = 20, 
                                 min_touches: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """サポート・レジスタンスライン検出"""
        try:
            values = data.values
            timestamps = data.timestamps
            
            if len(values) < window * 2:
                return {'support': [], 'resistance': []}
            
            # 局所最大・最小点を検出
            peaks = self._find_peaks(values, window)
            troughs = self._find_troughs(values, window)
            
            # サポートライン検出（安値の水平ライン）
            support_lines = self._find_horizontal_lines(
                troughs, values, timestamps, min_touches, line_type='support'
            )
            
            # レジスタンスライン検出（高値の水平ライン）
            resistance_lines = self._find_horizontal_lines(
                peaks, values, timestamps, min_touches, line_type='resistance'
            )
            
            return {
                'support': support_lines,
                'resistance': resistance_lines
            }
            
        except Exception as e:
            self.logger.error(f"Support/resistance detection failed: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """線形回帰の傾き計算"""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return 0.0
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x_squared = sum(x * x for x in x_values)
            
            denominator = n * sum_x_squared - sum_x * sum_x
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_intercept(self, x_values: List[float], y_values: List[float], slope: float) -> float:
        """線形回帰の切片計算"""
        try:
            if len(x_values) == 0 or len(y_values) == 0:
                return 0.0
            
            mean_x = statistics.mean(x_values)
            mean_y = statistics.mean(y_values)
            intercept = mean_y - slope * mean_x
            return intercept
            
        except Exception:
            return 0.0
    
    def _calculate_r_squared(self, x_values: List[float], y_values: List[float]) -> float:
        """決定係数（R²）計算"""
        try:
            if len(x_values) != len(y_values) or len(y_values) < 2:
                return 0.0
            
            # 線形回帰
            slope = self._calculate_slope(x_values, y_values)
            intercept = self._calculate_intercept(x_values, y_values, slope)
            
            # 予測値計算
            y_pred = [intercept + slope * x for x in x_values]
            
            # SSR（回帰平方和）とSST（全平方和）計算
            y_mean = statistics.mean(y_values)
            
            ss_res = sum((y_actual - y_predicted) ** 2 for y_actual, y_predicted in zip(y_values, y_pred))
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception:
            return 0.0
    
    def _calculate_trend_confidence(self, linear_trend: Dict[str, float], trend_strength: float) -> float:
        """トレンド分析の信頼度計算"""
        try:
            r_squared = linear_trend.get('r_squared', 0.0)
            
            # R²値とトレンド強度の組み合わせ
            base_confidence = (r_squared + trend_strength) / 2
            
            # 傾きの大きさも考慮
            slope = abs(linear_trend.get('slope', 0.0))
            slope_factor = min(slope / 0.1, 1.0)  # 傾きが0.1以上で最大評価
            
            confidence = base_confidence * (0.7 + 0.3 * slope_factor)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _find_peaks(self, values: List[float], window: int) -> List[int]:
        """ピーク（極大値）検出"""
        peaks = []
        
        for i in range(window, len(values) - window):
            is_peak = True
            current_value = values[i]
            
            # 周囲のウィンドウで最大値かチェック
            for j in range(i - window, i + window + 1):
                if j != i and values[j] >= current_value:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, values: List[float], window: int) -> List[int]:
        """谷（極小値）検出"""
        troughs = []
        
        for i in range(window, len(values) - window):
            is_trough = True
            current_value = values[i]
            
            # 周囲のウィンドウで最小値かチェック
            for j in range(i - window, i + window + 1):
                if j != i and values[j] <= current_value:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _find_horizontal_lines(self, points: List[int], values: List[float], 
                              timestamps: List[datetime], min_touches: int, 
                              line_type: str) -> List[Dict[str, Any]]:
        """水平ライン検出"""
        if len(points) < min_touches:
            return []
        
        lines = []
        tolerance = statistics.stdev(values) * 0.02 if len(values) > 1 else 0.01  # 2%の許容誤差
        
        for i, point_idx in enumerate(points):
            level = values[point_idx]
            touches = [point_idx]
            
            # 同じレベルの他のポイントを検索
            for j, other_idx in enumerate(points):
                if i != j and abs(values[other_idx] - level) <= tolerance:
                    touches.append(other_idx)
            
            if len(touches) >= min_touches:
                # 重複除去
                if not any(abs(line['level'] - level) <= tolerance for line in lines):
                    lines.append({
                        'type': line_type,
                        'level': level,
                        'touches': len(touches),
                        'touch_points': touches,
                        'touch_timestamps': [timestamps[idx] for idx in touches],
                        'strength': len(touches) / len(points)  # 強度
                    })
        
        # 強度順にソート
        lines.sort(key=lambda x: x['strength'], reverse=True)
        
        return lines[:5]  # 最大5本まで