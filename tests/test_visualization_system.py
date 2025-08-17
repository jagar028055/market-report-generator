"""
高度チャートビジュアライゼーションシステムテスト

予測チャート、リスクダッシュボード、インタラクティブチャート、
モンテカルロ可視化、3Dヒートマップの包括的テストスイート。
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
import sys
import os
from datetime import datetime, timedelta

# テスト対象モジュールのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.forecast_charts import ForecastChartGenerator
from visualization.risk_dashboard import RiskDashboard
from visualization.interactive_charts import InteractiveChartBuilder
from visualization.monte_carlo_viz import MonteCarloVisualizer
from visualization.risk_heatmap import RiskHeatmapGenerator


class TestForecastChartGenerator:
    """予測チャートジェネレーターテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.chart_gen = ForecastChartGenerator(chart_style='modern', color_palette='professional')
        
        # テストデータ準備
        np.random.seed(42)
        self.historical_data = np.cumsum(np.random.normal(0, 1, 50)) + 100
        self.forecast_data = np.cumsum(np.random.normal(0, 0.8, 10)) + self.historical_data[-1]
        self.upper_bound = self.forecast_data + 2
        self.lower_bound = self.forecast_data - 2
        
    def test_forecast_chart_generation(self):
        """基本予測チャート生成テスト"""
        data = {
            'historical_data': self.historical_data.tolist(),
            'forecast': self.forecast_data.tolist(),
            'upper_bound': self.upper_bound.tolist(),
            'lower_bound': self.lower_bound.tolist()
        }
        
        config = {'title': 'テスト予測チャート', 'x_label': '時間', 'y_label': '値'}
        result = self.chart_gen.generate_forecast_chart(data, config)
        
        assert 'chart_config' in result
        assert 'chart_id' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'forecast_basic'
        
        chart_config = result['chart_config']
        assert chart_config['type'] == 'line'
        assert len(chart_config['data']['datasets']) >= 2  # 実績＋予測
        assert 'labels' in chart_config['data']
        
    def test_model_comparison_chart(self):
        """モデル比較チャート生成テスト"""
        model_results = {
            'ARIMA': {
                'historical_data': self.historical_data.tolist(),
                'forecast': self.forecast_data.tolist()
            },
            'Random Forest': {
                'historical_data': self.historical_data.tolist(),
                'forecast': (self.forecast_data + np.random.normal(0, 0.5, len(self.forecast_data))).tolist()
            }
        }
        
        result = self.chart_gen.generate_model_comparison_chart(model_results)
        
        assert 'chart_config' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'model_comparison'
        
        chart_config = result['chart_config']
        assert len(chart_config['data']['datasets']) == 3  # 実績 + 2モデル
        
    def test_trend_decomposition_chart(self):
        """トレンド分解チャート生成テスト"""
        n = len(self.historical_data)
        decomposition_data = {
            'original': self.historical_data.tolist(),
            'trend': (np.linspace(98, 102, n)).tolist(),
            'seasonal': (2 * np.sin(2 * np.pi * np.arange(n) / 12)).tolist(),
            'residual': np.random.normal(0, 0.5, n).tolist()
        }
        
        result = self.chart_gen.generate_trend_decomposition_chart(decomposition_data)
        
        assert 'chart_configs' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'trend_decomposition'
        assert len(result['chart_configs']) == 4  # 4つのサブプロット
        
    def test_accuracy_metrics_chart(self):
        """精度メトリクスチャート生成テスト"""
        metrics_data = {
            'ARIMA': {'rmse': 2.5, 'mae': 1.8, 'mape': 5.2, 'r2': 0.85},
            'Random Forest': {'rmse': 2.1, 'mae': 1.5, 'mape': 4.8, 'r2': 0.89},
            'XGBoost': {'rmse': 1.9, 'mae': 1.4, 'mape': 4.3, 'r2': 0.91}
        }
        
        result = self.chart_gen.generate_accuracy_metrics_chart(metrics_data)
        
        assert 'chart_config' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'accuracy_metrics'
        
        chart_config = result['chart_config']
        assert chart_config['type'] == 'radar'
        assert len(chart_config['data']['datasets']) == 3
        
    def test_residual_analysis_chart(self):
        """残差分析チャート生成テスト"""
        n = 50
        fitted_values = np.random.normal(100, 10, n)
        residuals = np.random.normal(0, 2, n)
        
        residual_data = {
            'residuals': residuals.tolist(),
            'fitted_values': fitted_values.tolist()
        }
        
        result = self.chart_gen.generate_residual_analysis_chart(residual_data)
        
        assert 'chart_config' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'residual_analysis'
        
        chart_config = result['chart_config']
        assert chart_config['type'] == 'scatter'
        
    def test_html_template_generation(self):
        """HTMLテンプレート生成テスト"""
        # 複数チャート作成
        data = {
            'historical_data': self.historical_data.tolist(),
            'forecast': self.forecast_data.tolist()
        }
        
        charts = [
            self.chart_gen.generate_forecast_chart(data),
            self.chart_gen.generate_forecast_chart(data, {'title': 'チャート2'})
        ]
        
        html = self.chart_gen.generate_html_template(charts, layout='grid')
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'Chart.js' in html
        assert 'chart-grid' in html
        
    def test_color_palette_switching(self):
        """カラーパレット切り替えテスト"""
        # プロフェッショナル
        prof_gen = ForecastChartGenerator(color_palette='professional')
        prof_colors = prof_gen._get_colors()
        assert '#2E86AB' in prof_colors['primary']
        
        # バイブラント
        vib_gen = ForecastChartGenerator(color_palette='vibrant')
        vib_colors = vib_gen._get_colors()
        assert '#FF6B6B' in vib_colors['primary']
        
        # パステル
        pas_gen = ForecastChartGenerator(color_palette='pastel')
        pas_colors = pas_gen._get_colors()
        assert '#C7CEEA' in pas_colors['primary']


class TestRiskDashboard:
    """リスクダッシュボードテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.dashboard = RiskDashboard(dashboard_theme='dark', update_interval=5000)
        
        # テストVaRデータ
        self.var_data = {
            'var_95': 0.15,
            'var_99': 0.22,
            'cvar_95': 0.18
        }
        
        # テストストレス結果
        self.stress_results = {
            'financial_crisis': {
                'max_drawdown': 0.35,
                'scenario_volatility': 0.25,
                'prediction': [100, 95, 90, 85]
            },
            'high_inflation': {
                'max_drawdown': 0.20,
                'scenario_volatility': 0.18,
                'prediction': [100, 98, 96, 94]
            }
        }
        
    def test_var_dashboard_creation(self):
        """VaRダッシュボード作成テスト"""
        result = self.dashboard.create_var_dashboard(self.var_data)
        
        assert 'var_gauge' in result
        assert 'var_comparison' in result
        assert 'risk_level' in result
        assert 'risk_alerts' in result
        assert 'summary_metrics' in result
        
        # ゲージチャート確認
        gauge = result['var_gauge']
        assert gauge['type'] == 'doughnut'
        assert len(gauge['data']['datasets']) == 1
        
        # 比較チャート確認
        comparison = result['var_comparison']
        assert comparison['type'] == 'bar'
        assert len(comparison['data']['datasets']) == 1
        
    def test_stress_test_dashboard(self):
        """ストレステストダッシュボード作成テスト"""
        result = self.dashboard.create_stress_test_dashboard(self.stress_results)
        
        assert 'stress_comparison' in result
        assert 'worst_scenario' in result
        assert 'worst_drawdown' in result
        assert 'scenario_count' in result
        
        # レーダーチャート確認
        radar = result['stress_comparison']
        assert radar['type'] == 'radar'
        assert len(radar['data']['datasets']) == 2  # ドローダウン + ボラティリティ
        
    def test_correlation_dashboard(self):
        """相関ダッシュボード作成テスト"""
        correlation_data = {
            'correlation_matrix': {
                'Asset A': {'Asset A': 1.0, 'Asset B': 0.3, 'Asset C': -0.1},
                'Asset B': {'Asset A': 0.3, 'Asset B': 1.0, 'Asset C': 0.6},
                'Asset C': {'Asset A': -0.1, 'Asset B': 0.6, 'Asset C': 1.0}
            }
        }
        
        result = self.dashboard.create_correlation_dashboard(correlation_data)
        
        assert 'correlation_histogram' in result
        assert 'heatmap_data' in result
        assert 'high_correlations' in result
        assert 'average_correlation' in result
        
        # ヒストグラム確認
        histogram = result['correlation_histogram']
        assert histogram['type'] == 'bar'
        
    def test_portfolio_risk_dashboard(self):
        """ポートフォリオリスクダッシュボード作成テスト"""
        portfolio_data = {
            'risk_contributions': {
                'Asset A': 0.4,
                'Asset B': 0.35,
                'Asset C': 0.25
            },
            'risk_history': [0.12, 0.15, 0.14, 0.16, 0.13],
            'total_var': 0.14
        }
        
        result = self.dashboard.create_portfolio_risk_dashboard(portfolio_data)
        
        assert 'risk_pie' in result
        assert 'risk_timeline' in result
        assert 'concentration_risk' in result
        assert 'diversification_ratio' in result
        
        # 円グラフ確認
        pie = result['risk_pie']
        assert pie['type'] == 'pie'
        
        # タイムライン確認
        timeline = result['risk_timeline']
        assert timeline['type'] == 'line'
        
    def test_theme_switching(self):
        """テーマ切り替えテスト"""
        # ダークテーマ
        dark_dashboard = RiskDashboard(dashboard_theme='dark')
        dark_theme = dark_dashboard._get_theme()
        assert '#1e1e1e' in dark_theme['background']
        
        # ライトテーマ
        light_dashboard = RiskDashboard(dashboard_theme='light')
        light_theme = light_dashboard._get_theme()
        assert '#ffffff' in light_theme['background']
        
        # ブルーテーマ
        blue_dashboard = RiskDashboard(dashboard_theme='blue')
        blue_theme = blue_dashboard._get_theme()
        assert '#0f1419' in blue_theme['background']
        
    def test_dashboard_html_generation(self):
        """ダッシュボードHTML生成テスト"""
        dashboard_components = {
            'var_dashboard': self.dashboard.create_var_dashboard(self.var_data),
            'stress_dashboard': self.dashboard.create_stress_test_dashboard(self.stress_results)
        }
        
        html = self.dashboard.generate_dashboard_html(dashboard_components)
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'Chart.js' in html
        assert 'dashboard-container' in html


class TestInteractiveChartBuilder:
    """インタラクティブチャートビルダーテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.chart_builder = InteractiveChartBuilder(chart_library='plotly', theme='modern')
        
        # テスト時系列データ
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values1 = np.cumsum(np.random.normal(0, 1, 100)) + 100
        values2 = np.cumsum(np.random.normal(0, 0.8, 100)) + 50
        
        self.time_series_data = {
            'time_series': {
                'Series 1': {
                    'timestamps': dates.strftime('%Y-%m-%d').tolist(),
                    'values': values1.tolist()
                },
                'Series 2': {
                    'timestamps': dates.strftime('%Y-%m-%d').tolist(),
                    'values': values2.tolist()
                }
            },
            'confidence_bands': {
                'timestamps': dates[-20:].strftime('%Y-%m-%d').tolist(),
                'upper_bound': (values1[-20:] + 2).tolist(),
                'lower_bound': (values1[-20:] - 2).tolist()
            }
        }
        
    def test_interactive_timeseries_creation(self):
        """インタラクティブ時系列チャート作成テスト"""
        config = {'title': 'インタラクティブテストチャート'}
        result = self.chart_builder.create_interactive_timeseries(self.time_series_data, config)
        
        assert 'chart_config' in result
        assert 'chart_library' in result
        assert 'chart_type' in result
        assert result['chart_library'] == 'plotly'
        assert result['chart_type'] == 'interactive_timeseries'
        
        chart_config = result['chart_config']
        assert 'data' in chart_config
        assert 'layout' in chart_config
        assert len(chart_config['data']) >= 2  # 複数系列
        
    def test_multi_axis_chart(self):
        """多軸チャート作成テスト"""
        data = {
            'left_axis': {
                'Price': {
                    'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'values': [100, 105, 103]
                }
            },
            'right_axis': {
                'Volume': {
                    'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'values': [1000, 1200, 900]
                }
            }
        }
        
        config = {'title': '多軸テストチャート', 'left_y_label': '価格', 'right_y_label': '出来高'}
        result = self.chart_builder.create_multi_axis_chart(data, config)
        
        assert 'chart_config' in result
        assert result['chart_type'] == 'multi_axis'
        
        chart_config = result['chart_config']
        assert len(chart_config['data']) == 2
        assert chart_config['layout']['yaxis2']['side'] == 'right'
        
    def test_candlestick_chart(self):
        """ローソク足チャート作成テスト"""
        data = {
            'ohlc': {
                'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'open': [100, 105, 103],
                'high': [108, 110, 106],
                'low': [98, 103, 101],
                'close': [105, 103, 105]
            },
            'volume': {
                'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'values': [1000, 1200, 800]
            }
        }
        
        result = self.chart_builder.create_candlestick_chart(data)
        
        assert 'chart_config' in result
        assert result['chart_type'] == 'candlestick'
        
        chart_config = result['chart_config']
        assert len(chart_config['data']) == 2  # OHLC + 出来高
        assert chart_config['data'][0]['type'] == 'candlestick'
        
    def test_realtime_chart(self):
        """リアルタイムチャート作成テスト"""
        initial_data = {
            'time_series': {
                'Live Data': {
                    'timestamps': ['2024-01-01', '2024-01-02'],
                    'values': [100, 102]
                }
            }
        }
        
        config = {'update_interval': 3000}
        result = self.chart_builder.create_real_time_chart(initial_data, config)
        
        assert 'chart_config' in result
        assert 'realtime_js' in result
        assert 'update_interval' in result
        assert result['chart_type'] == 'realtime'
        assert result['update_interval'] == 3000
        
    def test_interactive_html_generation(self):
        """インタラクティブHTML生成テスト"""
        charts = [
            self.chart_builder.create_interactive_timeseries(self.time_series_data),
            self.chart_builder.create_multi_axis_chart({
                'left_axis': {'Series A': {'timestamps': ['T1'], 'values': [1]}},
                'right_axis': {'Series B': {'timestamps': ['T1'], 'values': [2]}}
            })
        ]
        
        html = self.chart_builder.generate_interactive_html(charts, layout='tabs')
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'plotly-latest.min.js' in html
        assert 'tab-button' in html


class TestMonteCarloVisualizer:
    """モンテカルロ可視化テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.visualizer = MonteCarloVisualizer(color_scheme='professional', animation=True)
        
        # テストシミュレーションデータ
        np.random.seed(42)
        self.simulations = np.random.normal(100, 10, (1000, 10))  # 1000回x10ステップ
        self.simulation_data = {
            'simulations': self.simulations,
            'base_prediction': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }
        
    def test_simulation_distribution_chart(self):
        """シミュレーション分布チャート作成テスト"""
        result = self.visualizer.create_simulation_distribution(self.simulation_data)
        
        assert 'chart_config' in result
        assert 'statistics' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'monte_carlo_distribution'
        
        chart_config = result['chart_config']
        assert len(chart_config['data']) == 2  # ヒストグラム + 正規分布
        
        # 統計確認
        stats = result['statistics']
        assert 'mean' in stats
        assert 'var_95' in stats
        assert 'var_99' in stats
        
    def test_risk_fan_chart(self):
        """リスクファンチャート作成テスト"""
        result = self.visualizer.create_risk_fan_chart(self.simulation_data)
        
        assert 'chart_config' in result
        assert 'percentile_data' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'risk_fan'
        
        chart_config = result['chart_config']
        assert len(chart_config['data']) >= 3  # 中央値 + 信頼区間
        
        # パーセンタイルデータ確認
        percentiles = result['percentile_data']
        assert 50 in percentiles  # 中央値
        assert 5 in percentiles   # VaR 95%
        
    def test_scenario_comparison(self):
        """シナリオ比較チャート作成テスト"""
        scenario_results = {
            'Baseline': {'simulations': np.random.normal(100, 8, (500, 10))},
            'Stress': {'simulations': np.random.normal(90, 15, (500, 10))},
            'Optimistic': {'simulations': np.random.normal(110, 5, (500, 10))}
        }
        
        result = self.visualizer.create_scenario_comparison(scenario_results)
        
        assert 'box_chart' in result
        assert 'violin_chart' in result
        assert 'comparison_stats' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'scenario_comparison'
        
        # ボックスプロット確認
        box_chart = result['box_chart']
        assert len(box_chart['data']) == 3  # 3シナリオ
        
        # 統計比較確認
        comp_stats = result['comparison_stats']
        assert 'Baseline' in comp_stats
        assert 'mean' in comp_stats['Baseline']
        
    def test_convergence_analysis(self):
        """収束分析チャート作成テスト"""
        result = self.visualizer.create_convergence_analysis(self.simulation_data)
        
        assert 'mean_convergence' in result
        assert 'std_convergence' in result
        assert 'convergence_metrics' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'convergence_analysis'
        
        # 収束メトリクス確認
        metrics = result['convergence_metrics']
        assert 'mean_error_percent' in metrics
        assert 'is_converged' in metrics
        
    def test_sensitivity_heatmap(self):
        """感度分析ヒートマップ作成テスト"""
        sensitivity_data = {
            'volatility': {
                'parameter_values': [0.1, 0.2, 0.3, 0.4, 0.5],
                'impact_values': [0.05, 0.15, 0.25, 0.40, 0.60]
            },
            'correlation': {
                'parameter_values': [0.2, 0.4, 0.6, 0.8],
                'impact_values': [0.10, 0.20, 0.35, 0.50]
            }
        }
        
        result = self.visualizer.create_sensitivity_heatmap(sensitivity_data)
        
        assert 'chart_config' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'sensitivity_heatmap'
        
        chart_config = result['chart_config']
        assert chart_config['data'][0]['type'] == 'heatmap'
        
    def test_comprehensive_report_generation(self):
        """包括的レポート生成テスト"""
        monte_carlo_results = {
            'monte_carlo': self.simulation_data,
            'stress_test': {
                'Crisis': {'prediction': [100, 95, 90], 'scenario_volatility': 0.2}
            },
            'sensitivity_analysis': {
                'param1': {'parameter_values': [1, 2, 3], 'impact_values': [0.1, 0.2, 0.3]}
            }
        }
        
        result = self.visualizer.generate_comprehensive_report(monte_carlo_results)
        
        assert 'charts' in result
        assert 'report_type' in result
        assert result['report_type'] == 'comprehensive_monte_carlo'
        
        charts = result['charts']
        assert 'distribution' in charts
        assert 'risk_fan' in charts
        
    def test_html_report_generation(self):
        """HTMLレポート生成テスト"""
        report_data = {
            'charts': {
                'distribution': self.visualizer.create_simulation_distribution(self.simulation_data)
            }
        }
        
        html = self.visualizer.generate_html_report(report_data)
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'plotly-latest.min.js' in html
        assert 'モンテカルロシミュレーション分析レポート' in html


class TestRiskHeatmapGenerator:
    """3Dリスクマップ・ヒートマップジェネレーターテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.heatmap_gen = RiskHeatmapGenerator(viz_style='3d_modern', color_intensity='high')
        
        # テスト相関データ
        self.correlation_data = {
            'correlation_matrix': {
                'Stock A': {'Stock A': 1.0, 'Stock B': 0.7, 'Stock C': 0.3},
                'Stock B': {'Stock A': 0.7, 'Stock B': 1.0, 'Stock C': 0.5},
                'Stock C': {'Stock A': 0.3, 'Stock B': 0.5, 'Stock C': 1.0}
            }
        }
        
        # テストリスクデータ
        self.risk_data = {
            'risks': [
                {'name': 'Market Risk', 'probability_index': 3, 'impact_index': 4},
                {'name': 'Credit Risk', 'probability_index': 2, 'impact_index': 3},
                {'name': 'Operational Risk', 'probability_index': 4, 'impact_index': 2}
            ]
        }
        
    def test_correlation_heatmap_creation(self):
        """相関ヒートマップ作成テスト"""
        result = self.heatmap_gen.create_correlation_heatmap(self.correlation_data)
        
        assert 'heatmap_2d' in result
        assert 'surface_3d' in result
        assert 'reordered_assets' in result
        assert 'cluster_info' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'correlation_heatmap'
        
        # 2Dヒートマップ確認
        heatmap_2d = result['heatmap_2d']
        assert heatmap_2d['data'][0]['type'] == 'heatmap'
        
        # 3Dサーフェス確認
        surface_3d = result['surface_3d']
        assert surface_3d['data'][0]['type'] == 'surface'
        
    def test_risk_matrix_heatmap(self):
        """リスクマトリックスヒートマップ作成テスト"""
        result = self.heatmap_gen.create_risk_matrix_heatmap(self.risk_data)
        
        assert 'heatmap_config' in result
        assert 'risk_details' in result
        assert 'risk_summary' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'risk_matrix'
        
        # ヒートマップ設定確認
        heatmap = result['heatmap_config']
        assert heatmap['data'][0]['type'] == 'heatmap'
        
        # リスクサマリー確認
        summary = result['risk_summary']
        assert 'total_risks' in summary
        assert summary['total_risks'] == 3
        
    def test_sector_risk_heatmap(self):
        """セクター別リスクヒートマップ作成テスト"""
        sector_data = {
            'Technology': {'volatility': 0.25, 'var_95': 0.15, 'max_drawdown': 0.20, 'correlation': 0.60},
            'Finance': {'volatility': 0.20, 'var_95': 0.12, 'max_drawdown': 0.18, 'correlation': 0.70},
            'Healthcare': {'volatility': 0.18, 'var_95': 0.10, 'max_drawdown': 0.15, 'correlation': 0.40}
        }
        
        result = self.heatmap_gen.create_sector_risk_heatmap(sector_data)
        
        assert 'heatmap_config' in result
        assert 'sector_summary' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'sector_risk_heatmap'
        
        # セクターサマリー確認
        summary = result['sector_summary']
        assert 'Technology' in summary
        assert 'risk_level' in summary['Technology']
        
    def test_temporal_risk_heatmap(self):
        """時間変化リスクヒートマップ作成テスト"""
        temporal_data = {
            'time_periods': ['Q1', 'Q2', 'Q3', 'Q4'],
            'assets': ['Asset A', 'Asset B', 'Asset C'],
            'risk_matrix': [
                [0.1, 0.2, 0.15],  # Q1
                [0.15, 0.25, 0.20], # Q2
                [0.20, 0.30, 0.25], # Q3
                [0.18, 0.28, 0.22]  # Q4
            ]
        }
        
        result = self.heatmap_gen.create_temporal_risk_heatmap(temporal_data)
        
        assert 'heatmap_2d' in result
        assert 'surface_3d' in result
        assert 'temporal_stats' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'temporal_risk_heatmap'
        
        # 時系列統計確認
        stats = result['temporal_stats']
        assert 'max_risk_period' in stats
        assert 'risk_trend' in stats
        
    def test_geographic_risk_map(self):
        """地理的リスクマップ作成テスト"""
        geographic_data = {
            'countries': ['USA', 'Japan', 'Germany'],
            'risk_scores': [0.15, 0.10, 0.12],
            'country_codes': ['USA', 'JPN', 'DEU']
        }
        
        result = self.heatmap_gen.create_geographic_risk_map(geographic_data)
        
        assert 'choropleth_map' in result
        assert 'geo_stats' in result
        assert 'chart_type' in result
        assert result['chart_type'] == 'geographic_risk_map'
        
        # コロプレスマップ確認
        choropleth = result['choropleth_map']
        assert choropleth['data'][0]['type'] == 'choropleth'
        
        # 地理統計確認
        stats = result['geo_stats']
        assert 'highest_risk_country' in stats
        assert 'avg_global_risk' in stats
        
    def test_comprehensive_heatmap_report(self):
        """包括的ヒートマップレポート生成テスト"""
        all_risk_data = {
            'correlation_data': self.correlation_data,
            'risk_matrix_data': self.risk_data,
            'sector_data': {
                'Tech': {'volatility': 0.3, 'var_95': 0.2},
                'Finance': {'volatility': 0.2, 'var_95': 0.15}
            }
        }
        
        result = self.heatmap_gen.generate_comprehensive_heatmap_report(all_risk_data)
        
        assert 'heatmaps' in result
        assert 'report_type' in result
        assert result['report_type'] == 'comprehensive_heatmap'
        
        heatmaps = result['heatmaps']
        assert 'correlation' in heatmaps
        assert 'risk_matrix' in heatmaps
        assert 'sector' in heatmaps
        
    def test_heatmap_html_generation(self):
        """ヒートマップHTML生成テスト"""
        heatmap_report = {
            'heatmaps': {
                'correlation': self.heatmap_gen.create_correlation_heatmap(self.correlation_data)
            },
            'viz_style': '3d_modern',
            'color_intensity': 'high'
        }
        
        html = self.heatmap_gen.generate_heatmap_html(heatmap_report)
        
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'plotly-latest.min.js' in html
        assert 'リスクヒートマップ分析レポート' in html


class TestVisualizationSystemIntegration:
    """可視化システム統合テスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.forecast_gen = ForecastChartGenerator()
        self.dashboard = RiskDashboard()
        self.chart_builder = InteractiveChartBuilder()
        self.mc_viz = MonteCarloVisualizer()
        self.heatmap_gen = RiskHeatmapGenerator()
        
    def test_visualization_pipeline(self):
        """可視化パイプラインテスト"""
        # 1. 予測データ準備
        forecast_data = {
            'historical_data': list(range(50)),
            'forecast': list(range(50, 60)),
            'upper_bound': list(range(52, 62)),
            'lower_bound': list(range(48, 58))
        }
        
        # 2. 予測チャート生成
        forecast_chart = self.forecast_gen.generate_forecast_chart(forecast_data)
        assert forecast_chart['chart_type'] == 'forecast_basic'
        
        # 3. リスクダッシュボード生成
        var_data = {'var_95': 0.15, 'var_99': 0.22, 'cvar_95': 0.18}
        var_dashboard = self.dashboard.create_var_dashboard(var_data)
        assert 'var_gauge' in var_dashboard
        
        # 4. モンテカルロ可視化
        mc_data = {'simulations': np.random.normal(100, 10, (100, 10))}
        mc_chart = self.mc_viz.create_simulation_distribution(mc_data)
        assert mc_chart['chart_type'] == 'monte_carlo_distribution'
        
        # 5. ヒートマップ生成
        corr_data = {
            'correlation_matrix': {
                'A': {'A': 1.0, 'B': 0.5},
                'B': {'A': 0.5, 'B': 1.0}
            }
        }
        heatmap = self.heatmap_gen.create_correlation_heatmap(corr_data)
        assert heatmap['chart_type'] == 'correlation_heatmap'
        
    def test_multi_chart_html_generation(self):
        """マルチチャートHTML生成テスト"""
        # 各種チャート作成
        charts = []
        
        # 予測チャート
        forecast_data = {'historical_data': [1, 2, 3], 'forecast': [4, 5, 6]}
        forecast_chart = self.forecast_gen.generate_forecast_chart(forecast_data)
        charts.append(forecast_chart)
        
        # インタラクティブチャート
        ts_data = {
            'time_series': {
                'Series1': {'timestamps': ['T1', 'T2', 'T3'], 'values': [1, 2, 3]}
            }
        }
        interactive_chart = self.chart_builder.create_interactive_timeseries(ts_data)
        charts.append(interactive_chart)
        
        # HTML生成テスト
        forecast_html = self.forecast_gen.generate_html_template(charts[:1])
        assert '<!DOCTYPE html>' in forecast_html
        
        interactive_html = self.chart_builder.generate_interactive_html(charts[1:])
        assert 'plotly-latest.min.js' in interactive_html
        
    def test_configuration_consistency(self):
        """設定一貫性テスト"""
        # 各コンポーネントの設定確認
        forecast_config = self.forecast_gen.get_chart_summary()
        assert 'available_charts' in forecast_config
        
        dashboard_config = self.dashboard.get_dashboard_config()
        assert 'available_components' in dashboard_config
        
        builder_config = self.chart_builder.get_builder_config()
        assert 'supported_chart_types' in builder_config
        
        viz_config = self.mc_viz.get_visualizer_config()
        assert 'supported_chart_types' in viz_config
        
        heatmap_config = self.heatmap_gen.get_generator_config()
        assert 'supported_heatmap_types' in heatmap_config
        
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 空データテスト
        empty_data = {}
        
        # 各コンポーネントが空データを適切に処理するか確認
        try:
            forecast_result = self.forecast_gen.generate_forecast_chart(empty_data)
            # 空の場合でもエラーにならずに基本構造を返すことを確認
            assert 'chart_config' in forecast_result
        except (ValueError, KeyError):
            # 適切なエラーが発生することも許容
            pass
            
        try:
            dashboard_result = self.dashboard.create_var_dashboard(empty_data)
            assert isinstance(dashboard_result, dict)
        except (ValueError, KeyError):
            pass
            
    def test_performance_characteristics(self):
        """パフォーマンス特性テスト"""
        import time
        
        # 大量データでのパフォーマンステスト
        large_data = {
            'historical_data': list(range(1000)),
            'forecast': list(range(1000, 1100))
        }
        
        start_time = time.time()
        result = self.forecast_gen.generate_forecast_chart(large_data)
        execution_time = time.time() - start_time
        
        # 1秒以内での処理を期待
        assert execution_time < 1.0
        assert 'chart_config' in result
        
        # モンテカルロ大量シミュレーション
        large_mc_data = {'simulations': np.random.normal(0, 1, (5000, 50))}
        
        start_time = time.time()
        mc_result = self.mc_viz.create_simulation_distribution(large_mc_data)
        mc_execution_time = time.time() - start_time
        
        # 5秒以内での処理を期待
        assert mc_execution_time < 5.0
        assert 'statistics' in mc_result


if __name__ == '__main__':
    # テスト実行
    import subprocess
    
    print("高度チャートビジュアライゼーションシステムテスト開始...")
    
    # pytest実行
    result = subprocess.run([
        'python', '-m', 'pytest', __file__, '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("エラー:", result.stderr)
        
    print(f"テスト終了: 終了コード={result.returncode}")