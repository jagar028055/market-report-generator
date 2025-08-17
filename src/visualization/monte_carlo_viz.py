"""
モンテカルロシミュレーション可視化

確率分布、リスクファン、信頼区間、パーセンタイル分析、
シナリオ比較を美しく可視化する包括的システム。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MonteCarloVisualizer:
    """
    モンテカルロシミュレーション可視化
    
    シミュレーション結果の分布、リスクファン、
    パーセンタイル分析、統計サマリーを可視化。
    """
    
    def __init__(self, color_scheme: str = 'professional', animation: bool = True):
        """
        Args:
            color_scheme: カラースキーム ('professional', 'risk_based', 'blue_gradient')
            animation: アニメーション有効フラグ
        """
        self.color_scheme = color_scheme
        self.animation = animation
        self.color_palettes = self._init_color_palettes()
        
    def _init_color_palettes(self) -> Dict[str, Dict[str, Any]]:
        """カラーパレット初期化"""
        return {
            'professional': {
                'background': '#ffffff',
                'grid': '#e6e6e6',
                'text': '#333333',
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'percentiles': [
                    '#d4e6f1', '#aed6f1', '#85c1e9', '#5dade2', '#3498db',
                    '#2e86ab', '#2874a6', '#21618c', '#1b4f72', '#154360'
                ]
            },
            'risk_based': {
                'background': '#f8f9fa',
                'grid': '#dee2e6',
                'text': '#495057',
                'primary': '#17a2b8',
                'secondary': '#6f42c1',
                'accent': '#fd7e14',
                'low_risk': '#28a745',
                'medium_risk': '#ffc107',
                'high_risk': '#dc3545',
                'percentiles': [
                    '#d1ecf1', '#b8daff', '#a2d2ff', '#7c3aed',
                    '#6366f1', '#4f46e5', '#4338ca', '#3730a3'
                ]
            },
            'blue_gradient': {
                'background': '#f0f8ff',
                'grid': '#b0c4de',
                'text': '#191970',
                'primary': '#4169e1',
                'secondary': '#1e90ff',
                'accent': '#00bfff',
                'percentiles': [
                    '#e6f3ff', '#ccebff', '#99d6ff', '#66c2ff',
                    '#33adff', '#0099ff', '#0080e6', '#0066cc'
                ]
            }
        }
        
    def _get_colors(self) -> Dict[str, Any]:
        """現在のカラーパレット取得"""
        return self.color_palettes.get(self.color_scheme, self.color_palettes['professional'])
        
    def create_simulation_distribution(self, simulation_data: Dict[str, Any],
                                     config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        シミュレーション分布チャート作成
        
        Args:
            simulation_data: シミュレーション結果
            config: チャート設定
            
        Returns:
            分布チャート設定
        """
        if config is None:
            config = {}
            
        colors = self._get_colors()
        
        # シミュレーションデータ抽出
        simulations = simulation_data.get('simulations', [])
        if isinstance(simulations, np.ndarray):
            # 最終ステップの値を使用
            final_values = simulations[:, -1] if simulations.ndim > 1 else simulations
        else:
            final_values = simulations
            
        # ヒストグラム分布
        hist_data, bin_edges = np.histogram(final_values, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 正規分布フィット
        mu, sigma = stats.norm.fit(final_values)
        normal_dist = stats.norm.pdf(bin_centers, mu, sigma)
        normal_dist_scaled = normal_dist * len(final_values) * (bin_edges[1] - bin_edges[0])
        
        # VaR線
        var_95 = np.percentile(final_values, 5)
        var_99 = np.percentile(final_values, 1)
        
        # チャート設定
        chart_config = {
            'data': [
                {
                    'x': bin_centers.tolist(),
                    'y': hist_data.tolist(),
                    'type': 'bar',
                    'name': 'シミュレーション分布',
                    'marker': {
                        'color': colors['primary'],
                        'opacity': 0.7
                    },
                    'hovertemplate': '値: %{x:.2f}<br>頻度: %{y}<extra></extra>'
                },
                {
                    'x': bin_centers.tolist(),
                    'y': normal_dist_scaled.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '正規分布フィット',
                    'line': {
                        'color': colors['secondary'],
                        'width': 3
                    },
                    'hovertemplate': '値: %{x:.2f}<br>密度: %{y:.4f}<extra></extra>'
                }
            ],
            'layout': {
                'title': {
                    'text': config.get('title', 'モンテカルロシミュレーション分布'),
                    'font': {'size': 18, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'xaxis': {
                    'title': config.get('x_label', '予測値'),
                    'gridcolor': colors['grid'],
                    'zeroline': False
                },
                'yaxis': {
                    'title': '頻度',
                    'gridcolor': colors['grid'],
                    'zeroline': False
                },
                'shapes': [
                    {
                        'type': 'line',
                        'x0': var_95, 'x1': var_95,
                        'y0': 0, 'y1': max(hist_data),
                        'line': {
                            'color': colors['danger'],
                            'width': 2,
                            'dash': 'dash'
                        }
                    },
                    {
                        'type': 'line',
                        'x0': var_99, 'x1': var_99,
                        'y0': 0, 'y1': max(hist_data),
                        'line': {
                            'color': colors['danger'],
                            'width': 2,
                            'dash': 'dot'
                        }
                    }
                ],
                'annotations': [
                    {
                        'x': var_95,
                        'y': max(hist_data) * 0.8,
                        'text': f'VaR 95%: {var_95:.2f}',
                        'showarrow': True,
                        'arrowhead': 2,
                        'arrowcolor': colors['danger']
                    },
                    {
                        'x': var_99,
                        'y': max(hist_data) * 0.6,
                        'text': f'VaR 99%: {var_99:.2f}',
                        'showarrow': True,
                        'arrowhead': 2,
                        'arrowcolor': colors['danger']
                    }
                ],
                'legend': {
                    'x': 0.7,
                    'y': 0.9
                }
            }
        }
        
        # 統計サマリー
        statistics = {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'skewness': stats.skew(final_values),
            'kurtosis': stats.kurtosis(final_values),
            'var_95': var_95,
            'var_99': var_99,
            'min': np.min(final_values),
            'max': np.max(final_values)
        }
        
        return {
            'chart_config': chart_config,
            'statistics': statistics,
            'chart_id': f'mc_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'monte_carlo_distribution'
        }
        
    def create_risk_fan_chart(self, simulation_data: Dict[str, Any],
                            config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        リスクファンチャート作成
        
        Args:
            simulation_data: 時系列シミュレーション結果
            config: チャート設定
            
        Returns:
            リスクファンチャート設定
        """
        if config is None:
            config = {}
            
        colors = self._get_colors()
        
        # シミュレーションデータ
        simulations = simulation_data.get('simulations', [])
        base_prediction = simulation_data.get('base_prediction', [])
        
        if isinstance(simulations, list):
            simulations = np.array(simulations)
            
        n_simulations, n_steps = simulations.shape
        
        # パーセンタイル計算
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_data = {}
        
        for step in range(n_steps):
            step_values = simulations[:, step]
            for p in percentiles:
                if p not in percentile_data:
                    percentile_data[p] = []
                percentile_data[p].append(np.percentile(step_values, p))
                
        # 時間軸
        time_labels = [f'T+{i+1}' for i in range(n_steps)]
        
        # リスクファン用データセット
        datasets = []
        percentile_colors = colors['percentiles']
        
        # 中央値
        datasets.append({
            'x': time_labels,
            'y': percentile_data[50],
            'type': 'scatter',
            'mode': 'lines',
            'name': '中央値 (50%)',
            'line': {
                'color': colors['primary'],
                'width': 3
            }
        })
        
        # 信頼区間（対称的に追加）
        confidence_bands = [
            (5, 95, '90%信頼区間'),
            (10, 90, '80%信頼区間'),
            (25, 75, '50%信頼区間')
        ]
        
        for i, (lower_p, upper_p, label) in enumerate(confidence_bands):
            color_alpha = 0.3 - i * 0.1
            fill_color = colors['primary'].replace(')', f', {color_alpha})').replace('rgb', 'rgba')
            
            # 上限
            datasets.append({
                'x': time_labels,
                'y': percentile_data[upper_p],
                'type': 'scatter',
                'mode': 'lines',
                'name': f'{label}上限',
                'line': {'color': 'rgba(0,0,0,0)'},
                'showlegend': False,
                'hoverinfo': 'skip'
            })
            
            # 下限（塗りつぶし）
            datasets.append({
                'x': time_labels,
                'y': percentile_data[lower_p],
                'type': 'scatter',
                'mode': 'lines',
                'name': label,
                'fill': 'tonexty',
                'fillcolor': fill_color,
                'line': {'color': colors['primary'], 'width': 1},
                'hovertemplate': f'{label}<br>時間: %{{x}}<br>値: %{{y:.2f}}<extra></extra>'
            })
            
        # ベース予測（あれば）
        if base_prediction:
            datasets.append({
                'x': time_labels,
                'y': base_prediction,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'ベース予測',
                'line': {
                    'color': colors['accent'],
                    'width': 2,
                    'dash': 'dash'
                },
                'marker': {'size': 4}
            })
            
        chart_config = {
            'data': datasets,
            'layout': {
                'title': {
                    'text': config.get('title', 'リスクファンチャート'),
                    'font': {'size': 18, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'xaxis': {
                    'title': '時間',
                    'gridcolor': colors['grid']
                },
                'yaxis': {
                    'title': config.get('y_label', '予測値'),
                    'gridcolor': colors['grid']
                },
                'legend': {
                    'x': 0.02,
                    'y': 0.98
                },
                'hovermode': 'x unified'
            }
        }
        
        return {
            'chart_config': chart_config,
            'percentile_data': percentile_data,
            'chart_id': f'risk_fan_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'risk_fan'
        }
        
    def create_scenario_comparison(self, scenario_results: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        シナリオ比較チャート作成
        
        Args:
            scenario_results: 複数シナリオ結果
            config: チャート設定
            
        Returns:
            シナリオ比較チャート設定
        """
        if config is None:
            config = {}
            
        colors = self._get_colors()
        
        # シナリオデータ整理
        scenarios = {}
        for scenario_name, data in scenario_results.items():
            if isinstance(data, dict) and 'simulations' in data:
                simulations = np.array(data['simulations'])
                final_values = simulations[:, -1] if simulations.ndim > 1 else simulations
                scenarios[scenario_name] = final_values
                
        # ボックスプロット用データ
        box_data = []
        scenario_names = list(scenarios.keys())
        color_list = [colors['primary'], colors['secondary'], colors['accent'], colors['success']]
        
        for i, (name, values) in enumerate(scenarios.items()):
            box_data.append({
                'y': values.tolist(),
                'type': 'box',
                'name': name,
                'boxpoints': 'outliers',
                'marker': {
                    'color': color_list[i % len(color_list)]
                },
                'line': {
                    'color': color_list[i % len(color_list)]
                }
            })
            
        # バイオリンプロット版
        violin_data = []
        for i, (name, values) in enumerate(scenarios.items()):
            violin_data.append({
                'y': values.tolist(),
                'type': 'violin',
                'name': name,
                'box': {'visible': True},
                'line': {'color': color_list[i % len(color_list)]},
                'fillcolor': color_list[i % len(color_list)].replace(')', ', 0.3)').replace('rgb', 'rgba'),
                'opacity': 0.6
            })
            
        # ボックスプロットチャート
        box_chart = {
            'data': box_data,
            'layout': {
                'title': {
                    'text': config.get('title', 'シナリオ比較（ボックスプロット）'),
                    'font': {'size': 16, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'yaxis': {
                    'title': '予測値',
                    'gridcolor': colors['grid']
                },
                'xaxis': {
                    'title': 'シナリオ',
                    'gridcolor': colors['grid']
                }
            }
        }
        
        # バイオリンプロットチャート
        violin_chart = {
            'data': violin_data,
            'layout': {
                'title': {
                    'text': config.get('title', 'シナリオ比較（分布形状）'),
                    'font': {'size': 16, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'yaxis': {
                    'title': '予測値',
                    'gridcolor': colors['grid']
                },
                'xaxis': {
                    'title': 'シナリオ',
                    'gridcolor': colors['grid']
                }
            }
        }
        
        # 統計比較テーブル
        comparison_stats = {}
        for name, values in scenarios.items():
            comparison_stats[name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'var_95': np.percentile(values, 5),
                'var_99': np.percentile(values, 1),
                'max_loss': np.min(values),
                'max_gain': np.max(values)
            }
            
        return {
            'box_chart': box_chart,
            'violin_chart': violin_chart,
            'comparison_stats': comparison_stats,
            'chart_id': f'scenario_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'scenario_comparison'
        }
        
    def create_convergence_analysis(self, simulation_data: Dict[str, Any],
                                  config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        収束分析チャート作成
        
        Args:
            simulation_data: シミュレーション結果
            config: チャート設定
            
        Returns:
            収束分析チャート設定
        """
        if config is None:
            config = {}
            
        colors = self._get_colors()
        
        simulations = np.array(simulation_data.get('simulations', []))
        n_simulations = len(simulations)
        
        # 段階的平均計算
        running_means = []
        running_stds = []
        sample_sizes = range(10, n_simulations + 1, max(1, n_simulations // 100))
        
        final_values = simulations[:, -1] if simulations.ndim > 1 else simulations
        
        for n in sample_sizes:
            subset = final_values[:n]
            running_means.append(np.mean(subset))
            running_stds.append(np.std(subset))
            
        # 理論値（全体平均）
        true_mean = np.mean(final_values)
        true_std = np.std(final_values)
        
        # 収束チャート
        convergence_chart = {
            'data': [
                {
                    'x': list(sample_sizes),
                    'y': running_means,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '累積平均',
                    'line': {
                        'color': colors['primary'],
                        'width': 2
                    }
                },
                {
                    'x': [sample_sizes[0], sample_sizes[-1]],
                    'y': [true_mean, true_mean],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '理論平均',
                    'line': {
                        'color': colors['danger'],
                        'width': 2,
                        'dash': 'dash'
                    }
                }
            ],
            'layout': {
                'title': {
                    'text': '平均値の収束',
                    'font': {'size': 16, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'xaxis': {
                    'title': 'サンプル数',
                    'gridcolor': colors['grid']
                },
                'yaxis': {
                    'title': '平均値',
                    'gridcolor': colors['grid']
                }
            }
        }
        
        # 標準偏差収束チャート
        std_convergence_chart = {
            'data': [
                {
                    'x': list(sample_sizes),
                    'y': running_stds,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '累積標準偏差',
                    'line': {
                        'color': colors['secondary'],
                        'width': 2
                    }
                },
                {
                    'x': [sample_sizes[0], sample_sizes[-1]],
                    'y': [true_std, true_std],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '理論標準偏差',
                    'line': {
                        'color': colors['danger'],
                        'width': 2,
                        'dash': 'dash'
                    }
                }
            ],
            'layout': {
                'title': {
                    'text': '標準偏差の収束',
                    'font': {'size': 16, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'xaxis': {
                    'title': 'サンプル数',
                    'gridcolor': colors['grid']
                },
                'yaxis': {
                    'title': '標準偏差',
                    'gridcolor': colors['grid']
                }
            }
        }
        
        # 収束度評価
        final_error_mean = abs(running_means[-1] - true_mean) / abs(true_mean) if true_mean != 0 else 0
        final_error_std = abs(running_stds[-1] - true_std) / abs(true_std) if true_std != 0 else 0
        
        convergence_metrics = {
            'mean_error_percent': final_error_mean * 100,
            'std_error_percent': final_error_std * 100,
            'is_converged': final_error_mean < 0.01 and final_error_std < 0.01,
            'recommended_samples': n_simulations if final_error_mean < 0.01 else n_simulations * 2
        }
        
        return {
            'mean_convergence': convergence_chart,
            'std_convergence': std_convergence_chart,
            'convergence_metrics': convergence_metrics,
            'chart_id': f'convergence_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'convergence_analysis'
        }
        
    def create_sensitivity_heatmap(self, sensitivity_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        感度分析ヒートマップ作成
        
        Args:
            sensitivity_data: 感度分析結果
            config: チャート設定
            
        Returns:
            ヒートマップチャート設定
        """
        if config is None:
            config = {}
            
        colors = self._get_colors()
        
        # 感度データ整理
        parameters = list(sensitivity_data.keys())
        param_values = []
        impact_values = []
        
        for param in parameters:
            param_data = sensitivity_data[param]
            param_values.append(param_data.get('parameter_values', []))
            impact_values.append(param_data.get('impact_values', []))
            
        # ヒートマップ用データマトリックス
        max_length = max(len(values) for values in param_values)
        impact_matrix = []
        
        for impacts in impact_values:
            # 長さを統一（パディング）
            padded_impacts = impacts + [0] * (max_length - len(impacts))
            impact_matrix.append(padded_impacts)
            
        # X軸ラベル（パラメータ値範囲）
        x_labels = [f'V{i+1}' for i in range(max_length)]
        
        heatmap_chart = {
            'data': [{
                'z': impact_matrix,
                'x': x_labels,
                'y': parameters,
                'type': 'heatmap',
                'colorscale': [
                    [0, colors['success']],
                    [0.5, '#ffffff'],
                    [1, colors['danger']]
                ],
                'colorbar': {
                    'title': '影響度',
                    'titlefont': {'color': colors['text']}
                },
                'hovetemplate': 'パラメータ: %{y}<br>値: %{x}<br>影響度: %{z:.3f}<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': config.get('title', '感度分析ヒートマップ'),
                    'font': {'size': 16, 'color': colors['text']}
                },
                'paper_bgcolor': colors['background'],
                'plot_bgcolor': colors['background'],
                'font': {'color': colors['text']},
                'xaxis': {
                    'title': 'パラメータ値',
                    'tickcolor': colors['text']
                },
                'yaxis': {
                    'title': 'パラメータ',
                    'tickcolor': colors['text']
                }
            }
        }
        
        return {
            'chart_config': heatmap_chart,
            'chart_id': f'sensitivity_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'sensitivity_heatmap'
        }
        
    def generate_comprehensive_report(self, monte_carlo_results: Dict[str, Any],
                                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        包括的モンテカルロレポート生成
        
        Args:
            monte_carlo_results: 全モンテカルロ結果
            config: レポート設定
            
        Returns:
            包括的レポート設定
        """
        if config is None:
            config = {}
            
        # 各種チャート作成
        charts = {}
        
        # 分布チャート
        if 'monte_carlo' in monte_carlo_results:
            charts['distribution'] = self.create_simulation_distribution(
                monte_carlo_results['monte_carlo'], 
                {'title': '予測値分布'}
            )
            
        # リスクファン
        if 'monte_carlo' in monte_carlo_results:
            charts['risk_fan'] = self.create_risk_fan_chart(
                monte_carlo_results['monte_carlo'],
                {'title': '予測リスクファン'}
            )
            
        # シナリオ比較
        if 'stress_test' in monte_carlo_results:
            scenario_data = {}
            for scenario, result in monte_carlo_results['stress_test'].items():
                if isinstance(result, dict) and 'prediction' in result:
                    # ダミーシミュレーションデータ生成
                    prediction = result['prediction']
                    volatility = result.get('scenario_volatility', 0.1)
                    dummy_sims = np.random.normal(prediction[-1], volatility, (1000, 1))
                    scenario_data[scenario] = {'simulations': dummy_sims}
                    
            if scenario_data:
                charts['scenario_comparison'] = self.create_scenario_comparison(
                    scenario_data,
                    {'title': 'ストレスシナリオ比較'}
                )
                
        # 収束分析
        if 'monte_carlo' in monte_carlo_results:
            charts['convergence'] = self.create_convergence_analysis(
                monte_carlo_results['monte_carlo'],
                {'title': 'シミュレーション収束性'}
            )
            
        # 感度分析
        if 'sensitivity_analysis' in monte_carlo_results:
            charts['sensitivity'] = self.create_sensitivity_heatmap(
                monte_carlo_results['sensitivity_analysis'],
                {'title': 'パラメータ感度分析'}
            )
            
        return {
            'charts': charts,
            'report_id': f'mc_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'report_type': 'comprehensive_monte_carlo',
            'generation_time': datetime.now().isoformat()
        }
        
    def generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        HTMLレポート生成
        
        Args:
            report_data: レポートデータ
            
        Returns:
            HTML文字列
        """
        colors = self._get_colors()
        charts = report_data.get('charts', {})
        
        # CSS
        css_styles = f"""
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: {colors['background']};
                color: {colors['text']};
            }}
            
            .report-header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
                color: white;
                border-radius: 10px;
            }}
            
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .chart-container {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border: 1px solid {colors['grid']};
            }}
            
            .chart-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: {colors['text']};
                border-bottom: 2px solid {colors['primary']};
                padding-bottom: 8px;
            }}
            
            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            
            .stats-table th, .stats-table td {{
                border: 1px solid {colors['grid']};
                padding: 8px 12px;
                text-align: left;
            }}
            
            .stats-table th {{
                background-color: {colors['primary']};
                color: white;
            }}
            
            .metric-highlight {{
                background-color: {colors['accent']};
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
        </style>
        """
        
        # チャートHTML生成
        chart_htmls = []
        chart_scripts = []
        
        for chart_name, chart_data in charts.items():
            chart_id = chart_data.get('chart_id', f'chart_{chart_name}')
            
            if chart_name == 'distribution':
                chart_html = f"""
                <div class="chart-container">
                    <div class="chart-title">予測値分布分析</div>
                    <div id="{chart_id}" style="width:100%;height:400px;"></div>
                    {self._generate_stats_table(chart_data.get('statistics', {}))}
                </div>
                """
                chart_htmls.append(chart_html)
                
                config = json.dumps(chart_data['chart_config'])
                chart_scripts.append(f"Plotly.newPlot('{chart_id}', {config}.data, {config}.layout);")
                
            elif chart_name == 'risk_fan':
                chart_html = f"""
                <div class="chart-container">
                    <div class="chart-title">リスクファン分析</div>
                    <div id="{chart_id}" style="width:100%;height:400px;"></div>
                </div>
                """
                chart_htmls.append(chart_html)
                
                config = json.dumps(chart_data['chart_config'])
                chart_scripts.append(f"Plotly.newPlot('{chart_id}', {config}.data, {config}.layout);")
                
            elif chart_name == 'scenario_comparison':
                box_id = f"{chart_id}_box"
                violin_id = f"{chart_id}_violin"
                
                chart_html = f"""
                <div class="chart-container">
                    <div class="chart-title">シナリオ比較分析</div>
                    <div id="{box_id}" style="width:100%;height:300px;"></div>
                    <div id="{violin_id}" style="width:100%;height:300px;"></div>
                    {self._generate_comparison_table(chart_data.get('comparison_stats', {}))}
                </div>
                """
                chart_htmls.append(chart_html)
                
                box_config = json.dumps(chart_data['box_chart'])
                violin_config = json.dumps(chart_data['violin_chart'])
                chart_scripts.append(f"Plotly.newPlot('{box_id}', {box_config}.data, {box_config}.layout);")
                chart_scripts.append(f"Plotly.newPlot('{violin_id}', {violin_config}.data, {violin_config}.layout);")
                
        # HTML構築
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>モンテカルロシミュレーションレポート</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {css_styles}
        </head>
        <body>
            <div class="report-header">
                <h1>モンテカルロシミュレーション分析レポート</h1>
                <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            </div>
            
            <div class="chart-grid">
                {''.join(chart_htmls)}
            </div>
            
            <script>
                {' '.join(chart_scripts)}
            </script>
        </body>
        </html>
        """
        
        return html_content
        
    def _generate_stats_table(self, statistics: Dict[str, float]) -> str:
        """統計テーブルHTML生成"""
        if not statistics:
            return ""
            
        table_rows = []
        for key, value in statistics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}"
                if 'var_' in key:
                    formatted_value = f'<span class="metric-highlight">{formatted_value}</span>'
                table_rows.append(f"<tr><td>{key}</td><td>{formatted_value}</td></tr>")
                
        return f"""
        <table class="stats-table">
            <tr><th>統計指標</th><th>値</th></tr>
            {''.join(table_rows)}
        </table>
        """
        
    def _generate_comparison_table(self, comparison_stats: Dict[str, Dict[str, float]]) -> str:
        """比較テーブルHTML生成"""
        if not comparison_stats:
            return ""
            
        headers = ['シナリオ', '平均', '中央値', '標準偏差', 'VaR 95%']
        table_rows = [f"<tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>"]
        
        for scenario, stats in comparison_stats.items():
            row_cells = [scenario]
            for metric in ['mean', 'median', 'std', 'var_95']:
                value = stats.get(metric, 0)
                row_cells.append(f"{value:.3f}")
                
            table_rows.append(f"<tr>{''.join(f'<td>{cell}</td>' for cell in row_cells)}</tr>")
            
        return f"""
        <table class="stats-table">
            {''.join(table_rows)}
        </table>
        """
        
    def get_visualizer_config(self) -> Dict[str, Any]:
        """可視化設定取得"""
        return {
            'color_scheme': self.color_scheme,
            'animation': self.animation,
            'available_color_schemes': list(self.color_palettes.keys()),
            'supported_chart_types': [
                'simulation_distribution',
                'risk_fan',
                'scenario_comparison',
                'convergence_analysis',
                'sensitivity_heatmap'
            ],
            'output_formats': ['plotly_json', 'html_report']
        }