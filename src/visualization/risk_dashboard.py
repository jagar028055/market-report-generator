"""
リスク分析ダッシュボード

VaR、ストレステスト、ポートフォリオリスク、相関分析を
統合したインタラクティブダッシュボード。リアルタイム
リスク監視、アラート機能、カスタマイズ可能な表示を提供。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskDashboard:
    """
    リスク分析ダッシュボード
    
    VaR分析、ストレステスト結果、リスク指標、
    相関マトリックス、リスクアラートを統合表示。
    """
    
    def __init__(self, dashboard_theme: str = 'dark', update_interval: int = 5000):
        """
        Args:
            dashboard_theme: テーマ ('dark', 'light', 'blue')
            update_interval: 更新間隔（ミリ秒）
        """
        self.dashboard_theme = dashboard_theme
        self.update_interval = update_interval
        self.theme_config = self._init_theme_config()
        self.risk_thresholds = self._init_risk_thresholds()
        
    def _init_theme_config(self) -> Dict[str, Dict[str, str]]:
        """テーマ設定初期化"""
        return {
            'dark': {
                'background': '#1e1e1e',
                'surface': '#2d2d2d',
                'primary': '#00d4ff',
                'secondary': '#ff6b6b',
                'accent': '#4ecdc4',
                'text_primary': '#ffffff',
                'text_secondary': '#cccccc',
                'border': '#404040',
                'success': '#51cf66',
                'warning': '#ffd43b',
                'danger': '#ff6b6b',
                'grid': '#404040'
            },
            'light': {
                'background': '#ffffff',
                'surface': '#f8f9fa',
                'primary': '#007bff',
                'secondary': '#6c757d',
                'accent': '#17a2b8',
                'text_primary': '#212529',
                'text_secondary': '#6c757d',
                'border': '#dee2e6',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'grid': '#e9ecef'
            },
            'blue': {
                'background': '#0f1419',
                'surface': '#1a2332',
                'primary': '#00b4d8',
                'secondary': '#90e0ef',
                'accent': '#caf0f8',
                'text_primary': '#ffffff',
                'text_secondary': '#b0b8c1',
                'border': '#2d3748',
                'success': '#68d391',
                'warning': '#f6e05e',
                'danger': '#fc8181',
                'grid': '#2d3748'
            }
        }
        
    def _init_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        """リスク閾値初期化"""
        return {
            'var': {
                'low': 0.05,      # 5%
                'medium': 0.10,   # 10% 
                'high': 0.20      # 20%
            },
            'volatility': {
                'low': 0.15,      # 15%
                'medium': 0.25,   # 25%
                'high': 0.40      # 40%
            },
            'correlation': {
                'low': 0.3,       # 30%
                'medium': 0.6,    # 60%
                'high': 0.8       # 80%
            },
            'drawdown': {
                'low': 0.10,      # 10%
                'medium': 0.20,   # 20%
                'high': 0.35      # 35%
            }
        }
        
    def _get_theme(self) -> Dict[str, str]:
        """現在のテーマ設定取得"""
        return self.theme_config.get(self.dashboard_theme, self.theme_config['dark'])
        
    def create_var_dashboard(self, var_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        VaRダッシュボード作成
        
        Args:
            var_data: VaR分析結果
            
        Returns:
            VaRダッシュボード設定
        """
        theme = self._get_theme()
        
        # VaR値とリスクレベル
        var_95 = abs(var_data.get('var_95', 0))
        var_99 = abs(var_data.get('var_99', 0))
        cvar_95 = abs(var_data.get('cvar_95', 0))
        
        # リスクレベル判定
        risk_level = self._assess_risk_level(var_95, 'var')
        risk_color = self._get_risk_color(risk_level, theme)
        
        # ゲージチャート設定
        var_gauge = {
            'type': 'doughnut',
            'data': {
                'labels': ['VaR 95%', '残余'],
                'datasets': [{
                    'data': [var_95 * 100, 100 - var_95 * 100],
                    'backgroundColor': [risk_color, theme['surface']],
                    'borderColor': [risk_color, theme['border']],
                    'borderWidth': 2
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'cutout': '70%',
                'plugins': {
                    'legend': {'display': False},
                    'tooltip': {
                        'callbacks': {
                            'label': 'function(context) { return context.label + ": " + context.parsed.toFixed(2) + "%"; }'
                        }
                    }
                }
            }
        }
        
        # VaR比較バーチャート
        var_comparison = {
            'type': 'bar',
            'data': {
                'labels': ['VaR 95%', 'VaR 99%', 'CVaR 95%'],
                'datasets': [{
                    'label': 'リスク値 (%)',
                    'data': [var_95 * 100, var_99 * 100, cvar_95 * 100],
                    'backgroundColor': [
                        theme['primary'],
                        theme['secondary'], 
                        theme['accent']
                    ],
                    'borderColor': theme['border'],
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': False},
                    'title': {
                        'display': True,
                        'text': 'VaR指標比較',
                        'color': theme['text_primary']
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    },
                    'x': {
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    }
                }
            }
        }
        
        # リスクアラート
        alerts = self._generate_risk_alerts(var_data, theme)
        
        return {
            'var_gauge': var_gauge,
            'var_comparison': var_comparison,
            'risk_level': risk_level,
            'risk_alerts': alerts,
            'summary_metrics': {
                'var_95_percent': f"{var_95:.2%}",
                'var_99_percent': f"{var_99:.2%}",
                'cvar_95_percent': f"{cvar_95:.2%}",
                'risk_level': risk_level
            }
        }
        
    def create_stress_test_dashboard(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ストレステストダッシュボード作成
        
        Args:
            stress_results: ストレステスト結果
            
        Returns:
            ストレステストダッシュボード設定
        """
        theme = self._get_theme()
        
        # シナリオ結果整理
        scenario_names = []
        max_drawdowns = []
        volatilities = []
        
        for scenario, result in stress_results.items():
            if isinstance(result, dict) and 'error' not in result:
                scenario_names.append(scenario)
                max_drawdowns.append(result.get('max_drawdown', 0) * 100)
                volatilities.append(result.get('scenario_volatility', 0))
                
        # ストレスシナリオ比較チャート
        stress_comparison = {
            'type': 'radar',
            'data': {
                'labels': scenario_names,
                'datasets': [
                    {
                        'label': '最大ドローダウン (%)',
                        'data': max_drawdowns,
                        'borderColor': theme['danger'],
                        'backgroundColor': theme['danger'].replace(')', ', 0.2)').replace('rgb', 'rgba'),
                        'borderWidth': 2,
                        'pointRadius': 4
                    },
                    {
                        'label': 'ボラティリティ',
                        'data': [v * 10 for v in volatilities],  # スケール調整
                        'borderColor': theme['warning'],
                        'backgroundColor': theme['warning'].replace(')', ', 0.2)').replace('rgb', 'rgba'),
                        'borderWidth': 2,
                        'pointRadius': 4
                    }
                ]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'ストレスシナリオ比較',
                        'color': theme['text_primary']
                    },
                    'legend': {
                        'labels': {'color': theme['text_secondary']}
                    }
                },
                'scales': {
                    'r': {
                        'beginAtZero': True,
                        'grid': {'color': theme['grid']},
                        'pointLabels': {'color': theme['text_secondary']},
                        'ticks': {'color': theme['text_secondary']}
                    }
                }
            }
        }
        
        # 最悪シナリオ特定
        worst_scenario = scenario_names[max_drawdowns.index(max(max_drawdowns))] if max_drawdowns else 'N/A'
        worst_drawdown = max(max_drawdowns) if max_drawdowns else 0
        
        return {
            'stress_comparison': stress_comparison,
            'worst_scenario': worst_scenario,
            'worst_drawdown': f"{worst_drawdown:.1f}%",
            'scenario_count': len(scenario_names),
            'average_drawdown': f"{np.mean(max_drawdowns):.1f}%" if max_drawdowns else "N/A"
        }
        
    def create_correlation_dashboard(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        相関分析ダッシュボード作成
        
        Args:
            correlation_data: 相関分析結果
            
        Returns:
            相関ダッシュボード設定
        """
        theme = self._get_theme()
        
        # 相関マトリックス準備
        correlations = correlation_data.get('correlation_matrix', {})
        assets = list(correlations.keys())
        
        # ヒートマップデータ
        heatmap_data = []
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                correlation_value = correlations.get(asset1, {}).get(asset2, 0)
                heatmap_data.append({
                    'x': asset2,
                    'y': asset1,
                    'v': correlation_value
                })
                
        # 高相関ペア検出
        high_correlations = []
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    corr = correlations.get(asset1, {}).get(asset2, 0)
                    if abs(corr) > self.risk_thresholds['correlation']['medium']:
                        high_correlations.append({
                            'pair': f"{asset1}-{asset2}",
                            'correlation': corr
                        })
                        
        # 相関分布ヒストグラム
        all_correlations = []
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    corr = correlations.get(asset1, {}).get(asset2, 0)
                    all_correlations.append(abs(corr))
                    
        # ヒストグラムビン
        hist_bins = np.linspace(0, 1, 11)
        hist_counts, _ = np.histogram(all_correlations, bins=hist_bins)
        
        correlation_histogram = {
            'type': 'bar',
            'data': {
                'labels': [f"{hist_bins[i]:.1f}-{hist_bins[i+1]:.1f}" for i in range(len(hist_counts))],
                'datasets': [{
                    'label': '相関分布',
                    'data': hist_counts.tolist(),
                    'backgroundColor': theme['primary'],
                    'borderColor': theme['border'],
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': '相関係数分布',
                        'color': theme['text_primary']
                    },
                    'legend': {'display': False}
                },
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    },
                    'x': {
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    }
                }
            }
        }
        
        return {
            'correlation_histogram': correlation_histogram,
            'heatmap_data': heatmap_data,
            'high_correlations': high_correlations[:5],  # 上位5件
            'average_correlation': f"{np.mean(all_correlations):.3f}" if all_correlations else "N/A",
            'max_correlation': f"{max(all_correlations):.3f}" if all_correlations else "N/A"
        }
        
    def create_portfolio_risk_dashboard(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ポートフォリオリスクダッシュボード作成
        
        Args:
            portfolio_data: ポートフォリオリスクデータ
            
        Returns:
            ポートフォリオダッシュボード設定
        """
        theme = self._get_theme()
        
        # リスク貢献度
        risk_contributions = portfolio_data.get('risk_contributions', {})
        assets = list(risk_contributions.keys())
        contributions = list(risk_contributions.values())
        
        # リスク貢献度円グラフ
        risk_pie = {
            'type': 'pie',
            'data': {
                'labels': assets,
                'datasets': [{
                    'data': contributions,
                    'backgroundColor': [
                        theme['primary'], theme['secondary'], theme['accent'],
                        theme['success'], theme['warning'], theme['danger']
                    ][:len(assets)],
                    'borderColor': theme['border'],
                    'borderWidth': 2
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'リスク貢献度',
                        'color': theme['text_primary']
                    },
                    'legend': {
                        'position': 'bottom',
                        'labels': {'color': theme['text_secondary']}
                    }
                }
            }
        }
        
        # 時系列リスク推移
        risk_history = portfolio_data.get('risk_history', [])
        time_labels = [f"T-{len(risk_history)-i-1}" for i in range(len(risk_history))]
        
        risk_timeline = {
            'type': 'line',
            'data': {
                'labels': time_labels,
                'datasets': [{
                    'label': 'ポートフォリオVaR',
                    'data': risk_history,
                    'borderColor': theme['danger'],
                    'backgroundColor': 'transparent',
                    'borderWidth': 3,
                    'pointRadius': 2,
                    'tension': 0.2
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'リスク推移',
                        'color': theme['text_primary']
                    },
                    'legend': {'display': False}
                },
                'scales': {
                    'y': {
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    },
                    'x': {
                        'ticks': {'color': theme['text_secondary']},
                        'grid': {'color': theme['grid']}
                    }
                }
            }
        }
        
        # 集中度リスク
        concentration_risk = max(contributions) if contributions else 0
        diversification_ratio = 1 - concentration_risk
        
        return {
            'risk_pie': risk_pie,
            'risk_timeline': risk_timeline,
            'concentration_risk': f"{concentration_risk:.1%}",
            'diversification_ratio': f"{diversification_ratio:.1%}",
            'total_portfolio_var': portfolio_data.get('total_var', 'N/A'),
            'top_risk_contributor': assets[contributions.index(max(contributions))] if contributions else 'N/A'
        }
        
    def _assess_risk_level(self, value: float, risk_type: str) -> str:
        """リスクレベル評価"""
        thresholds = self.risk_thresholds.get(risk_type, {})
        
        if value <= thresholds.get('low', 0):
            return 'low'
        elif value <= thresholds.get('medium', 0):
            return 'medium'
        else:
            return 'high'
            
    def _get_risk_color(self, risk_level: str, theme: Dict[str, str]) -> str:
        """リスクレベル色取得"""
        color_map = {
            'low': theme['success'],
            'medium': theme['warning'],
            'high': theme['danger']
        }
        return color_map.get(risk_level, theme['secondary'])
        
    def _generate_risk_alerts(self, risk_data: Dict[str, Any], 
                            theme: Dict[str, str]) -> List[Dict[str, str]]:
        """リスクアラート生成"""
        alerts = []
        
        # VaRアラート
        var_95 = abs(risk_data.get('var_95', 0))
        if var_95 > self.risk_thresholds['var']['high']:
            alerts.append({
                'type': 'danger',
                'title': '高リスクアラート',
                'message': f'VaR 95%が{var_95:.1%}と高水準です',
                'color': theme['danger']
            })
        elif var_95 > self.risk_thresholds['var']['medium']:
            alerts.append({
                'type': 'warning',
                'title': '中リスクアラート',
                'message': f'VaR 95%が{var_95:.1%}です',
                'color': theme['warning']
            })
            
        # その他リスクアラート
        if 'stress_test' in risk_data:
            stress_results = risk_data['stress_test']
            for scenario, result in stress_results.items():
                if isinstance(result, dict):
                    max_dd = result.get('max_drawdown', 0)
                    if max_dd > self.risk_thresholds['drawdown']['high']:
                        alerts.append({
                            'type': 'danger',
                            'title': 'ストレステストアラート',
                            'message': f'{scenario}シナリオで{max_dd:.1%}のドローダウン',
                            'color': theme['danger']
                        })
                        
        return alerts
        
    def generate_dashboard_html(self, dashboard_components: Dict[str, Any]) -> str:
        """
        ダッシュボードHTML生成
        
        Args:
            dashboard_components: ダッシュボードコンポーネント
            
        Returns:
            HTML文字列
        """
        theme = self._get_theme()
        
        # CSS
        css_styles = f"""
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                background: {theme['background']};
                color: {theme['text_primary']};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
            }}
            
            .dashboard-container {{
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            
            .dashboard-header {{
                background: {theme['surface']};
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid {theme['border']};
            }}
            
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .dashboard-card {{
                background: {theme['surface']};
                border: 1px solid {theme['border']};
                border-radius: 12px;
                padding: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            .dashboard-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            .card-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: {theme['text_primary']};
                border-bottom: 2px solid {theme['primary']};
                padding-bottom: 8px;
            }}
            
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: {theme['primary']};
                margin: 10px 0;
            }}
            
            .alert-item {{
                background: rgba(255, 107, 107, 0.1);
                border-left: 4px solid;
                padding: 12px;
                margin: 8px 0;
                border-radius: 4px;
            }}
            
            .alert-danger {{
                border-left-color: {theme['danger']};
                background: {theme['danger'].replace(')', ', 0.1)').replace('rgb', 'rgba')};
            }}
            
            .alert-warning {{
                border-left-color: {theme['warning']};
                background: {theme['warning'].replace(')', ', 0.1)').replace('rgb', 'rgba')};
            }}
            
            .chart-container {{
                position: relative;
                height: 300px;
                margin: 15px 0;
            }}
            
            .status-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            
            .status-low {{ background: {theme['success']}; }}
            .status-medium {{ background: {theme['warning']}; }}
            .status-high {{ background: {theme['danger']}; }}
            
            .update-time {{
                color: {theme['text_secondary']};
                font-size: 12px;
                text-align: right;
                margin-top: 10px;
            }}
        </style>
        """
        
        # JavaScript
        charts_js = self._generate_dashboard_js(dashboard_components)
        
        # HTML構築
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>リスク分析ダッシュボード</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {css_styles}
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>リスク分析ダッシュボード</h1>
                    <div class="update-time">最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                
                <div class="dashboard-grid">
                    {self._generate_dashboard_cards(dashboard_components, theme)}
                </div>
            </div>
            
            {charts_js}
        </body>
        </html>
        """
        
        return html_content
        
    def _generate_dashboard_cards(self, components: Dict[str, Any], 
                                theme: Dict[str, str]) -> str:
        """ダッシュボードカード生成"""
        cards_html = []
        
        # VaRカード
        if 'var_dashboard' in components:
            var_data = components['var_dashboard']
            risk_level = var_data.get('risk_level', 'low')
            
            var_card = f"""
            <div class="dashboard-card">
                <div class="card-title">
                    <span class="status-indicator status-{risk_level}"></span>
                    Value at Risk (VaR)
                </div>
                <div class="chart-container">
                    <canvas id="var_gauge_chart"></canvas>
                </div>
                <div class="metric-value">{var_data.get('summary_metrics', {}).get('var_95_percent', 'N/A')}</div>
                <div>VaR 95%</div>
                
                <div style="margin-top: 15px;">
                    {self._generate_alerts_html(var_data.get('risk_alerts', []))}
                </div>
            </div>
            """
            cards_html.append(var_card)
            
        # ストレステストカード
        if 'stress_dashboard' in components:
            stress_data = components['stress_dashboard']
            
            stress_card = f"""
            <div class="dashboard-card">
                <div class="card-title">ストレステスト</div>
                <div class="chart-container">
                    <canvas id="stress_comparison_chart"></canvas>
                </div>
                <div>最悪シナリオ: <strong>{stress_data.get('worst_scenario', 'N/A')}</strong></div>
                <div>最大ドローダウン: <span class="metric-value">{stress_data.get('worst_drawdown', 'N/A')}</span></div>
            </div>
            """
            cards_html.append(stress_card)
            
        # 相関分析カード
        if 'correlation_dashboard' in components:
            corr_data = components['correlation_dashboard']
            
            correlation_card = f"""
            <div class="dashboard-card">
                <div class="card-title">相関分析</div>
                <div class="chart-container">
                    <canvas id="correlation_histogram_chart"></canvas>
                </div>
                <div>平均相関: <span class="metric-value">{corr_data.get('average_correlation', 'N/A')}</span></div>
                <div>最大相関: <span class="metric-value">{corr_data.get('max_correlation', 'N/A')}</span></div>
            </div>
            """
            cards_html.append(correlation_card)
            
        return '\n'.join(cards_html)
        
    def _generate_alerts_html(self, alerts: List[Dict[str, str]]) -> str:
        """アラートHTML生成"""
        if not alerts:
            return '<div style="color: #28a745;">リスクアラートなし</div>'
            
        alerts_html = []
        for alert in alerts:
            alert_class = f"alert-{alert['type']}"
            alert_html = f"""
            <div class="alert-item {alert_class}">
                <strong>{alert['title']}</strong><br>
                {alert['message']}
            </div>
            """
            alerts_html.append(alert_html)
            
        return '\n'.join(alerts_html)
        
    def _generate_dashboard_js(self, components: Dict[str, Any]) -> str:
        """ダッシュボードJavaScript生成"""
        js_scripts = ['<script>']
        
        # 各チャートのJS生成
        for component_name, component_data in components.items():
            if component_name == 'var_dashboard':
                if 'var_gauge' in component_data:
                    config = json.dumps(component_data['var_gauge'])
                    js_scripts.append(f"""
                    const varGaugeCtx = document.getElementById('var_gauge_chart');
                    if (varGaugeCtx) {{
                        new Chart(varGaugeCtx, {config});
                    }}
                    """)
                    
            elif component_name == 'stress_dashboard':
                if 'stress_comparison' in component_data:
                    config = json.dumps(component_data['stress_comparison'])
                    js_scripts.append(f"""
                    const stressCtx = document.getElementById('stress_comparison_chart');
                    if (stressCtx) {{
                        new Chart(stressCtx, {config});
                    }}
                    """)
                    
            elif component_name == 'correlation_dashboard':
                if 'correlation_histogram' in component_data:
                    config = json.dumps(component_data['correlation_histogram'])
                    js_scripts.append(f"""
                    const corrCtx = document.getElementById('correlation_histogram_chart');
                    if (corrCtx) {{
                        new Chart(corrCtx, {config});
                    }}
                    """)
                    
        # 自動更新機能
        js_scripts.append(f"""
        // 自動更新（{self.update_interval}ms間隔）
        setInterval(function() {{
            // ここで新しいデータを取得してチャート更新
            console.log('Dashboard update check...');
        }}, {self.update_interval});
        """)
        
        js_scripts.append('</script>')
        return '\n'.join(js_scripts)
        
    def get_dashboard_config(self) -> Dict[str, Any]:
        """ダッシュボード設定取得"""
        return {
            'theme': self.dashboard_theme,
            'update_interval': self.update_interval,
            'risk_thresholds': self.risk_thresholds,
            'available_components': [
                'var_dashboard',
                'stress_dashboard', 
                'correlation_dashboard',
                'portfolio_dashboard'
            ],
            'supported_themes': list(self.theme_config.keys())
        }