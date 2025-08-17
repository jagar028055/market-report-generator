"""
Time Series Analyzer

時系列分析の統合インターフェース
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field

# 基本的な統計・数値計算（標準ライブラリ使用）
import statistics
from collections import defaultdict, deque

# 依存ライブラリ（条件付きインポート）
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# サブモジュールのインポート
from .trend_analysis import TrendAnalyzer
from .volatility import VolatilityAnalyzer
from .correlation import CorrelationAnalyzer
from .seasonal import SeasonalAnalyzer


@dataclass
class TimeSeriesData:
    """時系列データ構造"""
    timestamps: List[datetime]
    values: List[float]
    name: str = "TimeSeries"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """データ検証"""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have same length")
        
        if len(self.timestamps) == 0:
            raise ValueError("Empty time series data")


@dataclass
class AnalysisResult:
    """分析結果"""
    analysis_type: str
    result_data: Dict[str, Any]
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TimeSeriesAnalyzer:
    """
    時系列分析統合クラス
    
    Features:
    - トレンド分析（移動平均、指数平滑化）
    - ボラティリティ分析（標準偏差、VaR）
    - 相関分析（ピアソン、スピアマン）
    - 季節性分析（周期検出、季節調整）
    - 統計サマリー（基本統計量、分布）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        分析器初期化
        
        Args:
            config: 分析設定
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # サブアナライザー初期化
        self.trend_analyzer = TrendAnalyzer(self.config.get('trend', {}))
        self.volatility_analyzer = VolatilityAnalyzer(self.config.get('volatility', {}))
        self.correlation_analyzer = CorrelationAnalyzer(self.config.get('correlation', {}))
        self.seasonal_analyzer = SeasonalAnalyzer(self.config.get('seasonal', {}))
        
        # 分析履歴
        self.analysis_history: List[AnalysisResult] = []
        
        self.logger.info("TimeSeriesAnalyzer initialized")
    
    def analyze_comprehensive(self, data: TimeSeriesData, 
                            analysis_types: Optional[List[str]] = None) -> Dict[str, AnalysisResult]:
        """
        包括的時系列分析
        
        Args:
            data: 時系列データ
            analysis_types: 実行する分析タイプのリスト
            
        Returns:
            分析結果辞書
        """
        try:
            if analysis_types is None:
                analysis_types = ['trend', 'volatility', 'statistics', 'correlation', 'seasonal']
            
            results = {}
            
            # 基本統計
            if 'statistics' in analysis_types:
                results['statistics'] = self.calculate_basic_statistics(data)
            
            # トレンド分析
            if 'trend' in analysis_types:
                results['trend'] = self.trend_analyzer.analyze_trend(data)
            
            # ボラティリティ分析
            if 'volatility' in analysis_types:
                results['volatility'] = self.volatility_analyzer.analyze_volatility(data)
            
            # 相関分析（他データとの比較が必要な場合）
            if 'correlation' in analysis_types:
                # 自己相関分析
                results['autocorrelation'] = self.correlation_analyzer.analyze_autocorrelation(data)
            
            # 季節性分析
            if 'seasonal' in analysis_types:
                results['seasonal'] = self.seasonal_analyzer.analyze_seasonality(data)
            
            # 結果を履歴に追加
            for analysis_type, result in results.items():
                self.analysis_history.append(result)
            
            self.logger.info(f"Comprehensive analysis completed for {data.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise
    
    def calculate_basic_statistics(self, data: TimeSeriesData) -> AnalysisResult:
        """基本統計量計算"""
        try:
            values = data.values
            
            # 基本統計量
            stats = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            }
            
            # 分位数計算
            if NUMPY_AVAILABLE:
                np_values = np.array(values)
                stats.update({
                    'q25': float(np.percentile(np_values, 25)),
                    'q75': float(np.percentile(np_values, 75)),
                    'skewness': self._calculate_skewness(values),
                    'kurtosis': self._calculate_kurtosis(values)
                })
            else:
                # 基本実装
                sorted_values = sorted(values)
                n = len(sorted_values)
                stats.update({
                    'q25': sorted_values[n // 4],
                    'q75': sorted_values[3 * n // 4],
                    'skewness': self._calculate_skewness(values),
                    'kurtosis': self._calculate_kurtosis(values)
                })
            
            # 変化率計算
            if len(values) > 1:
                changes = []
                for i in range(1, len(values)):
                    if values[i-1] != 0:
                        change = (values[i] - values[i-1]) / values[i-1] * 100
                        changes.append(change)
                
                if changes:
                    stats.update({
                        'mean_change_pct': statistics.mean(changes),
                        'std_change_pct': statistics.stdev(changes) if len(changes) > 1 else 0.0,
                        'max_gain_pct': max(changes),
                        'max_loss_pct': min(changes)
                    })
            
            return AnalysisResult(
                analysis_type='statistics',
                result_data=stats,
                confidence=1.0,
                metadata={'data_points': len(values)}
            )
            
        except Exception as e:
            self.logger.error(f"Basic statistics calculation failed: {str(e)}")
            raise
    
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
    
    def detect_outliers(self, data: TimeSeriesData, 
                       method: str = 'iqr') -> AnalysisResult:
        """外れ値検出"""
        try:
            values = data.values
            outliers = []
            outlier_indices = []
            
            if method == 'iqr':
                # IQR法
                if NUMPY_AVAILABLE:
                    np_values = np.array(values)
                    q1 = np.percentile(np_values, 25)
                    q3 = np.percentile(np_values, 75)
                else:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    q1 = sorted_values[n // 4]
                    q3 = sorted_values[3 * n // 4]
                
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                for i, value in enumerate(values):
                    if value < lower_bound or value > upper_bound:
                        outliers.append(value)
                        outlier_indices.append(i)
            
            elif method == 'zscore':
                # Z-score法
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 1.0
                
                for i, value in enumerate(values):
                    z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                    if z_score > 3:  # 3σ ルール
                        outliers.append(value)
                        outlier_indices.append(i)
            
            result_data = {
                'method': method,
                'outlier_count': len(outliers),
                'outlier_ratio': len(outliers) / len(values),
                'outlier_values': outliers,
                'outlier_indices': outlier_indices,
                'outlier_timestamps': [data.timestamps[i] for i in outlier_indices]
            }
            
            return AnalysisResult(
                analysis_type='outlier_detection',
                result_data=result_data,
                confidence=0.95 if method == 'iqr' else 0.99,
                metadata={'method': method, 'total_points': len(values)}
            )
            
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {str(e)}")
            raise
    
    def calculate_returns(self, data: TimeSeriesData, 
                         return_type: str = 'simple') -> TimeSeriesData:
        """リターン計算"""
        try:
            values = data.values
            
            if len(values) < 2:
                raise ValueError("Need at least 2 data points to calculate returns")
            
            returns = []
            return_timestamps = data.timestamps[1:]  # 最初の時点は除外
            
            for i in range(1, len(values)):
                if return_type == 'simple':
                    # 単純リターン
                    if values[i-1] != 0:
                        ret = (values[i] - values[i-1]) / values[i-1]
                    else:
                        ret = 0.0
                elif return_type == 'log':
                    # 対数リターン
                    if values[i] > 0 and values[i-1] > 0:
                        ret = math.log(values[i] / values[i-1])
                    else:
                        ret = 0.0
                else:
                    raise ValueError(f"Unknown return type: {return_type}")
                
                returns.append(ret)
            
            return TimeSeriesData(
                timestamps=return_timestamps,
                values=returns,
                name=f"{data.name}_returns_{return_type}",
                metadata={
                    'return_type': return_type,
                    'original_series': data.name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Return calculation failed: {str(e)}")
            raise
    
    def calculate_rolling_statistics(self, data: TimeSeriesData, 
                                   window: int = 20,
                                   statistics_types: Optional[List[str]] = None) -> Dict[str, TimeSeriesData]:
        """移動統計量計算"""
        try:
            if statistics_types is None:
                statistics_types = ['mean', 'std', 'min', 'max']
            
            values = data.values
            timestamps = data.timestamps
            
            if len(values) < window:
                raise ValueError(f"Data length ({len(values)}) < window size ({window})")
            
            results = {}
            
            for stat_type in statistics_types:
                rolling_values = []
                rolling_timestamps = []
                
                for i in range(window - 1, len(values)):
                    window_data = values[i - window + 1:i + 1]
                    
                    if stat_type == 'mean':
                        stat_value = statistics.mean(window_data)
                    elif stat_type == 'std':
                        stat_value = statistics.stdev(window_data) if len(window_data) > 1 else 0.0
                    elif stat_type == 'min':
                        stat_value = min(window_data)
                    elif stat_type == 'max':
                        stat_value = max(window_data)
                    elif stat_type == 'median':
                        stat_value = statistics.median(window_data)
                    else:
                        continue
                    
                    rolling_values.append(stat_value)
                    rolling_timestamps.append(timestamps[i])
                
                results[stat_type] = TimeSeriesData(
                    timestamps=rolling_timestamps,
                    values=rolling_values,
                    name=f"{data.name}_rolling_{stat_type}_{window}",
                    metadata={'window': window, 'statistic': stat_type}
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Rolling statistics calculation failed: {str(e)}")
            raise
    
    def get_analysis_summary(self, data: TimeSeriesData) -> Dict[str, Any]:
        """分析サマリー取得"""
        try:
            # 基本統計
            basic_stats = self.calculate_basic_statistics(data)
            
            # 直近トレンド
            recent_trend = self.trend_analyzer.detect_trend_direction(data, window=20)
            
            # ボラティリティレベル
            volatility = self.volatility_analyzer.calculate_rolling_volatility(data, window=20)
            current_volatility = volatility.values[-1] if volatility.values else 0.0
            
            # 外れ値検出
            outliers = self.detect_outliers(data)
            
            summary = {
                'data_info': {
                    'name': data.name,
                    'start_date': data.timestamps[0].isoformat(),
                    'end_date': data.timestamps[-1].isoformat(),
                    'data_points': len(data.values),
                    'frequency': self._estimate_frequency(data.timestamps)
                },
                'basic_statistics': basic_stats.result_data,
                'current_trend': recent_trend,
                'current_volatility': current_volatility,
                'outlier_ratio': outliers.result_data['outlier_ratio'],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Analysis summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _estimate_frequency(self, timestamps: List[datetime]) -> str:
        """データ頻度推定"""
        try:
            if len(timestamps) < 2:
                return 'unknown'
            
            # 隣接する時点の差分を計算
            deltas = []
            for i in range(1, min(10, len(timestamps))):  # 最初の10個を確認
                delta = timestamps[i] - timestamps[i-1]
                deltas.append(delta.total_seconds())
            
            avg_delta = statistics.mean(deltas)
            
            # 頻度判定
            if avg_delta <= 60:  # 1分以下
                return 'minute'
            elif avg_delta <= 3600:  # 1時間以下
                return 'hourly'
            elif avg_delta <= 86400:  # 1日以下
                return 'daily'
            elif avg_delta <= 604800:  # 1週間以下
                return 'weekly'
            elif avg_delta <= 2678400:  # 1ヶ月以下
                return 'monthly'
            else:
                return 'yearly'
                
        except Exception:
            return 'unknown'
    
    def export_analysis_results(self, results: Dict[str, AnalysisResult], 
                               format: str = 'dict') -> Union[Dict, str]:
        """分析結果エクスポート"""
        try:
            if format == 'dict':
                export_data = {}
                for analysis_type, result in results.items():
                    export_data[analysis_type] = {
                        'type': result.analysis_type,
                        'data': result.result_data,
                        'confidence': result.confidence,
                        'timestamp': result.timestamp.isoformat(),
                        'metadata': result.metadata
                    }
                return export_data
            
            elif format == 'json':
                import json
                export_data = self.export_analysis_results(results, 'dict')
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Analysis results export failed: {str(e)}")
            raise