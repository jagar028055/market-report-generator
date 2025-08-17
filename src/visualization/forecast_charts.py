"""
予測結果可視化チャート

ARIMA、機械学習、アンサンブル予測の結果を美しく可視化。
信頼区間、モデル比較、トレンド分析、季節性分解を含む
包括的な予測チャートシステム。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import base64
from io import BytesIO
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ForecastChartGenerator:
    """
    予測結果可視化チャートジェネレーター
    
    予測モデル結果の可視化、信頼区間表示、モデル比較チャート、
    トレンド分析、季節性分解チャートを生成。
    """
    
    def __init__(self, chart_style: str = 'modern', color_palette: str = 'professional'):
        """
        Args:
            chart_style: チャートスタイル ('modern', 'classic', 'minimal')
            color_palette: カラーパレット ('professional', 'vibrant', 'pastel')
        """
        self.chart_style = chart_style
        self.color_palette = color_palette
        self.color_schemes = self._init_color_schemes()
        
    def _init_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """カラーパレット初期化"""
        return {
            'professional': {
                'primary': '#2E86AB',
                'secondary': '#A23B72', 
                'accent': '#F18F01',
                'success': '#C73E1D',
                'background': '#F5F5F5',
                'grid': '#E0E0E0',
                'text': '#333333',
                'confidence': 'rgba(46, 134, 171, 0.2)',
                'forecast': '#FF6B35'
            },
            'vibrant': {
                'primary': '#FF6B6B',
                'secondary': '#4ECDC4',
                'accent': '#45B7D1',
                'success': '#96CEB4',
                'background': '#FFEAA7',
                'grid': '#DDD',
                'text': '#2D3436',
                'confidence': 'rgba(255, 107, 107, 0.2)',
                'forecast': '#A29BFE'
            },
            'pastel': {
                'primary': '#C7CEEA',
                'secondary': '#FFB7B2',
                'accent': '#FFDAC1',
                'success': '#E2F0CB',
                'background': '#FFFFFF',
                'grid': '#F0F0F0',
                'text': '#555555',
                'confidence': 'rgba(199, 206, 234, 0.3)',
                'forecast': '#B2B2FF'
            }
        }
        
    def _get_colors(self) -> Dict[str, str]:
        """現在のカラーパレット取得"""
        return self.color_schemes.get(self.color_palette, self.color_schemes['professional'])
        
    def generate_forecast_chart(self, data: Dict[str, Any], 
                              chart_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        基本予測チャート生成
        
        Args:
            data: 予測データ（実績値、予測値、信頼区間）
            chart_config: チャート設定
            
        Returns:
            チャート設定とHTML
        """
        if chart_config is None:
            chart_config = {}
            
        colors = self._get_colors()
        
        # データ準備
        historical_data = data.get('historical_data', [])
        forecast_data = data.get('forecast', [])
        upper_bound = data.get('upper_bound', [])
        lower_bound = data.get('lower_bound', [])
        
        # 時間軸生成
        hist_labels = self._generate_time_labels(len(historical_data), 'historical')
        forecast_labels = self._generate_time_labels(len(forecast_data), 'forecast', 
                                                   start_from=len(historical_data))
        
        # Chart.js設定
        chart_config_js = {
            'type': 'line',
            'data': {
                'labels': hist_labels + forecast_labels,
                'datasets': [
                    {
                        'label': '実績値',
                        'data': historical_data + [None] * len(forecast_data),
                        'borderColor': colors['primary'],
                        'backgroundColor': 'transparent',
                        'borderWidth': 2,
                        'pointRadius': 3,
                        'pointHoverRadius': 5,
                        'tension': 0.2
                    },
                    {
                        'label': '予測値',
                        'data': [None] * (len(historical_data) - 1) + [historical_data[-1]] + forecast_data,
                        'borderColor': colors['forecast'],
                        'backgroundColor': 'transparent',
                        'borderWidth': 2,
                        'borderDash': [5, 5],
                        'pointRadius': 3,
                        'pointHoverRadius': 5,
                        'tension': 0.2
                    }
                ]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': chart_config.get('title', '予測チャート'),
                        'font': {'size': 16, 'weight': 'bold'},
                        'color': colors['text']
                    },
                    'legend': {
                        'display': True,
                        'position': 'top',
                        'labels': {'color': colors['text']}
                    },
                    'tooltip': {
                        'mode': 'index',
                        'intersect': False,
                        'backgroundColor': 'rgba(0,0,0,0.8)',
                        'titleColor': '#FFFFFF',
                        'bodyColor': '#FFFFFF'
                    }
                },
                'scales': {
                    'x': {
                        'display': True,
                        'title': {
                            'display': True,
                            'text': chart_config.get('x_label', '時間'),
                            'color': colors['text']
                        },
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    },
                    'y': {
                        'display': True,
                        'title': {
                            'display': True,
                            'text': chart_config.get('y_label', '値'),
                            'color': colors['text']
                        },
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    }
                },
                'interaction': {
                    'mode': 'nearest',
                    'axis': 'x',
                    'intersect': False
                }
            }
        }
        
        # 信頼区間追加
        if upper_bound and lower_bound:
            chart_config_js['data']['datasets'].append({
                'label': '信頼区間',
                'data': [None] * (len(historical_data) - 1) + [historical_data[-1]] + upper_bound,
                'borderColor': colors['confidence'].replace('0.2', '0.5'),
                'backgroundColor': colors['confidence'],
                'fill': '+1',
                'pointRadius': 0,
                'borderWidth': 1,
                'tension': 0.2
            })
            chart_config_js['data']['datasets'].append({
                'label': '',
                'data': [None] * (len(historical_data) - 1) + [historical_data[-1]] + lower_bound,
                'borderColor': colors['confidence'].replace('0.2', '0.5'),
                'backgroundColor': colors['confidence'],
                'fill': False,
                'pointRadius': 0,
                'borderWidth': 1,
                'tension': 0.2,
                'showLine': True
            })
            
        return {
            'chart_config': chart_config_js,
            'chart_id': f'forecast_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'forecast_basic'
        }
        
    def generate_model_comparison_chart(self, model_results: Dict[str, Dict[str, Any]],
                                      chart_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        複数モデル比較チャート生成
        
        Args:
            model_results: {'model_name': {'forecast': [...], 'metrics': {...}}}
            chart_config: チャート設定
            
        Returns:
            比較チャート設定
        """
        if chart_config is None:
            chart_config = {}
            
        colors = self._get_colors()
        color_list = [colors['primary'], colors['secondary'], colors['accent'], colors['success']]
        
        # 実績データ（最初のモデルから取得）
        first_model = next(iter(model_results.values()))
        historical_data = first_model.get('historical_data', [])
        
        # 時間軸
        max_forecast_length = max(len(result.get('forecast', [])) for result in model_results.values())
        hist_labels = self._generate_time_labels(len(historical_data), 'historical')
        forecast_labels = self._generate_time_labels(max_forecast_length, 'forecast',
                                                   start_from=len(historical_data))
        
        # データセット構築
        datasets = [
            {
                'label': '実績値',
                'data': historical_data + [None] * max_forecast_length,
                'borderColor': '#000000',
                'backgroundColor': 'transparent',
                'borderWidth': 3,
                'pointRadius': 2,
                'tension': 0.2
            }
        ]
        
        # 各モデルの予測を追加
        for i, (model_name, result) in enumerate(model_results.items()):
            forecast = result.get('forecast', [])
            color = color_list[i % len(color_list)]
            
            # 予測データを最大長に合わせる
            padded_forecast = forecast + [None] * (max_forecast_length - len(forecast))
            
            datasets.append({
                'label': f'{model_name} 予測',
                'data': [None] * (len(historical_data) - 1) + [historical_data[-1] if historical_data else 0] + padded_forecast,
                'borderColor': color,
                'backgroundColor': 'transparent',
                'borderWidth': 2,
                'borderDash': [3, 3] if i > 0 else [],
                'pointRadius': 3,
                'tension': 0.2
            })
            
        chart_config_js = {
            'type': 'line',
            'data': {
                'labels': hist_labels + forecast_labels,
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': chart_config.get('title', 'モデル比較チャート'),
                        'font': {'size': 16, 'weight': 'bold'},
                        'color': colors['text']
                    },
                    'legend': {
                        'display': True,
                        'position': 'top',
                        'labels': {'color': colors['text']}
                    }
                },
                'scales': {
                    'x': {
                        'title': {'display': True, 'text': '時間', 'color': colors['text']},
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    },
                    'y': {
                        'title': {'display': True, 'text': '値', 'color': colors['text']},
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    }
                }
            }
        }
        
        return {
            'chart_config': chart_config_js,
            'chart_id': f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'model_comparison'
        }
        
    def generate_trend_decomposition_chart(self, decomposition_data: Dict[str, Any],
                                         chart_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        トレンド分解チャート生成
        
        Args:
            decomposition_data: 分解データ（original, trend, seasonal, residual）
            chart_config: チャート設定
            
        Returns:
            分解チャート設定
        """
        if chart_config is None:
            chart_config = {}
            
        colors = self._get_colors()
        
        # データ準備
        original = decomposition_data.get('original', [])
        trend = decomposition_data.get('trend', [])
        seasonal = decomposition_data.get('seasonal', [])
        residual = decomposition_data.get('residual', [])
        
        time_labels = self._generate_time_labels(len(original), 'historical')
        
        # 4つのサブプロット用データ
        subplots = [
            {
                'title': '元データ',
                'data': original,
                'color': colors['primary']
            },
            {
                'title': 'トレンド成分',
                'data': trend,
                'color': colors['secondary']
            },
            {
                'title': '季節成分',
                'data': seasonal,
                'color': colors['accent']
            },
            {
                'title': '残差成分',
                'data': residual,
                'color': colors['success']
            }
        ]
        
        # 複数チャート設定
        charts = []
        for i, subplot in enumerate(subplots):
            chart_config_js = {
                'type': 'line',
                'data': {
                    'labels': time_labels,
                    'datasets': [{
                        'label': subplot['title'],
                        'data': subplot['data'],
                        'borderColor': subplot['color'],
                        'backgroundColor': 'transparent',
                        'borderWidth': 2,
                        'pointRadius': 1,
                        'tension': 0.2
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': subplot['title'],
                            'font': {'size': 14, 'weight': 'bold'},
                            'color': colors['text']
                        },
                        'legend': {'display': False}
                    },
                    'scales': {
                        'x': {
                            'display': i == 3,  # 最下段のみX軸表示
                            'grid': {'color': colors['grid']},
                            'ticks': {'color': colors['text']}
                        },
                        'y': {
                            'grid': {'color': colors['grid']},
                            'ticks': {'color': colors['text']}
                        }
                    }
                }
            }
            charts.append(chart_config_js)
            
        return {
            'chart_configs': charts,
            'chart_id': f'decomposition_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'trend_decomposition',
            'layout': 'vertical_stack'
        }
        
    def generate_accuracy_metrics_chart(self, metrics_data: Dict[str, Dict[str, float]],
                                      chart_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        予測精度メトリクスチャート生成
        
        Args:
            metrics_data: {'model_name': {'rmse': ..., 'mae': ..., 'r2': ...}}
            chart_config: チャート設定
            
        Returns:
            メトリクスチャート設定
        """
        if chart_config is None:
            chart_config = {}
            
        colors = self._get_colors()
        
        # モデル名とメトリクス
        model_names = list(metrics_data.keys())
        metrics = ['rmse', 'mae', 'mape', 'r2']
        
        # レーダーチャート用データ準備
        datasets = []
        color_list = [colors['primary'], colors['secondary'], colors['accent'], colors['success']]
        
        for i, model_name in enumerate(model_names):
            model_metrics = metrics_data[model_name]
            
            # メトリクス値正規化（R²以外は逆数化して大きいほど良いに統一）
            normalized_values = []
            for metric in metrics:
                value = model_metrics.get(metric, 0)
                if metric == 'r2':
                    normalized_values.append(max(0, min(1, value)) * 100)  # 0-100%
                elif metric in ['rmse', 'mae', 'mape']:
                    # 逆数化（小さいほど良い → 大きいほど良い）
                    if value > 0:
                        normalized_values.append(min(100, 100 / (1 + value)))
                    else:
                        normalized_values.append(100)
                else:
                    normalized_values.append(value)
                    
            color = color_list[i % len(color_list)]
            datasets.append({
                'label': model_name,
                'data': normalized_values,
                'borderColor': color,
                'backgroundColor': color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                'borderWidth': 2,
                'pointRadius': 4,
                'pointHoverRadius': 6
            })
            
        chart_config_js = {
            'type': 'radar',
            'data': {
                'labels': ['RMSE', 'MAE', 'MAPE', 'R²'],
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': chart_config.get('title', 'モデル精度比較'),
                        'font': {'size': 16, 'weight': 'bold'},
                        'color': colors['text']
                    },
                    'legend': {
                        'display': True,
                        'position': 'top',
                        'labels': {'color': colors['text']}
                    }
                },
                'scales': {
                    'r': {
                        'beginAtZero': True,
                        'max': 100,
                        'grid': {'color': colors['grid']},
                        'pointLabels': {'color': colors['text']},
                        'ticks': {'color': colors['text']}
                    }
                }
            }
        }
        
        return {
            'chart_config': chart_config_js,
            'chart_id': f'accuracy_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'accuracy_metrics'
        }
        
    def generate_residual_analysis_chart(self, residual_data: Dict[str, Any],
                                       chart_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        残差分析チャート生成
        
        Args:
            residual_data: 残差データ（residuals, fitted_values）
            chart_config: チャート設定
            
        Returns:
            残差分析チャート設定
        """
        if chart_config is None:
            chart_config = {}
            
        colors = self._get_colors()
        
        residuals = residual_data.get('residuals', [])
        fitted_values = residual_data.get('fitted_values', [])
        
        # 散布図チャート
        scatter_data = []
        for i, (fitted, residual) in enumerate(zip(fitted_values, residuals)):
            scatter_data.append({'x': fitted, 'y': residual})
            
        chart_config_js = {
            'type': 'scatter',
            'data': {
                'datasets': [
                    {
                        'label': '残差',
                        'data': scatter_data,
                        'backgroundColor': colors['primary'],
                        'borderColor': colors['primary'],
                        'pointRadius': 4,
                        'pointHoverRadius': 6
                    },
                    {
                        'label': 'ゼロライン',
                        'data': [
                            {'x': min(fitted_values) if fitted_values else 0, 'y': 0},
                            {'x': max(fitted_values) if fitted_values else 1, 'y': 0}
                        ],
                        'type': 'line',
                        'borderColor': colors['grid'],
                        'borderWidth': 2,
                        'borderDash': [5, 5],
                        'pointRadius': 0,
                        'fill': False
                    }
                ]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': chart_config.get('title', '残差分析'),
                        'font': {'size': 16, 'weight': 'bold'},
                        'color': colors['text']
                    },
                    'legend': {
                        'display': True,
                        'labels': {'color': colors['text']}
                    }
                },
                'scales': {
                    'x': {
                        'title': {'display': True, 'text': '予測値', 'color': colors['text']},
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    },
                    'y': {
                        'title': {'display': True, 'text': '残差', 'color': colors['text']},
                        'grid': {'color': colors['grid']},
                        'ticks': {'color': colors['text']}
                    }
                }
            }
        }
        
        return {
            'chart_config': chart_config_js,
            'chart_id': f'residual_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'residual_analysis'
        }
        
    def _generate_time_labels(self, length: int, data_type: str, 
                            start_from: int = 0) -> List[str]:
        """時間軸ラベル生成"""
        labels = []
        
        if data_type == 'historical':
            for i in range(length):
                labels.append(f'T-{length-i-1}')
        elif data_type == 'forecast':
            for i in range(length):
                labels.append(f'T+{i+1}')
        else:
            for i in range(length):
                labels.append(f'期間{start_from + i + 1}')
                
        return labels
        
    def generate_html_template(self, charts: List[Dict[str, Any]], 
                             layout: str = 'grid') -> str:
        """
        チャート表示HTMLテンプレート生成
        
        Args:
            charts: チャート設定リスト
            layout: レイアウト ('grid', 'vertical', 'horizontal')
            
        Returns:
            HTML文字列
        """
        colors = self._get_colors()
        
        # CSSスタイル
        css_style = f"""
        <style>
            .forecast-chart-container {{
                background-color: {colors['background']};
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .chart-vertical {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            
            .chart-horizontal {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .chart-item {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                min-height: 400px;
            }}
            
            .chart-title {{
                color: {colors['text']};
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                text-align: center;
            }}
            
            canvas {{
                max-height: 400px !important;
            }}
        </style>
        """
        
        # JavaScript
        chart_scripts = []
        for chart in charts:
            chart_id = chart['chart_id']
            config = json.dumps(chart['chart_config'])
            
            script = f"""
            (function() {{
                const canvas = document.getElementById('{chart_id}');
                if (canvas) {{
                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {config});
                }}
            }})();
            """
            chart_scripts.append(script)
            
        # HTML構築
        layout_class = f'chart-{layout}'
        chart_html_items = []
        
        for chart in charts:
            chart_id = chart['chart_id']
            chart_type = chart.get('chart_type', 'forecast')
            
            item_html = f"""
            <div class="chart-item">
                <canvas id="{chart_id}" width="400" height="300"></canvas>
            </div>
            """
            chart_html_items.append(item_html)
            
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>予測チャート</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {css_style}
        </head>
        <body>
            <div class="forecast-chart-container">
                <div class="{layout_class}">
                    {''.join(chart_html_items)}
                </div>
            </div>
            
            <script>
                {' '.join(chart_scripts)}
            </script>
        </body>
        </html>
        """
        
        return html_template
        
    def export_chart_config(self, chart_data: Dict[str, Any]) -> str:
        """チャート設定JSON出力"""
        return json.dumps(chart_data, indent=2, ensure_ascii=False)
        
    def get_chart_summary(self) -> Dict[str, Any]:
        """チャート生成サマリー"""
        return {
            'chart_style': self.chart_style,
            'color_palette': self.color_palette,
            'available_charts': [
                'forecast_basic',
                'model_comparison', 
                'trend_decomposition',
                'accuracy_metrics',
                'residual_analysis'
            ],
            'supported_formats': ['html', 'json'],
            'chart_library': 'Chart.js'
        }