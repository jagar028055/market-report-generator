"""
インタラクティブ時系列チャート

D3.js、Plotly.jsを活用した高度なインタラクティブチャート。
ズーム、パン、データ探索、リアルタイム更新、
カスタムアノテーション機能を提供。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class InteractiveChartBuilder:
    """
    インタラクティブチャートビルダー
    
    高度なインタラクション機能を持つ時系列チャート、
    多軸表示、カスタムツールチップ、アニメーション効果を提供。
    """
    
    def __init__(self, chart_library: str = 'plotly', theme: str = 'modern'):
        """
        Args:
            chart_library: チャートライブラリ ('plotly', 'd3', 'highcharts')
            theme: チャートテーマ ('modern', 'classic', 'dark')
        """
        self.chart_library = chart_library
        self.theme = theme
        self.theme_config = self._init_theme_config()
        
    def _init_theme_config(self) -> Dict[str, Dict[str, Any]]:
        """テーマ設定初期化"""
        return {
            'modern': {
                'background': '#ffffff',
                'grid_color': '#e1e5e9',
                'text_color': '#2c3e50',
                'primary_color': '#3498db',
                'secondary_color': '#e74c3c',
                'accent_color': '#f39c12',
                'font_family': "'Segoe UI', sans-serif",
                'font_size': 12
            },
            'classic': {
                'background': '#f8f9fa',
                'grid_color': '#dee2e6',
                'text_color': '#495057',
                'primary_color': '#007bff',
                'secondary_color': '#dc3545',
                'accent_color': '#ffc107',
                'font_family': "'Times New Roman', serif",
                'font_size': 11
            },
            'dark': {
                'background': '#2c3e50',
                'grid_color': '#34495e',
                'text_color': '#ecf0f1',
                'primary_color': '#3498db',
                'secondary_color': '#e74c3c',
                'accent_color': '#f1c40f',
                'font_family': "'Roboto', sans-serif",
                'font_size': 12
            }
        }
        
    def _get_theme(self) -> Dict[str, Any]:
        """現在のテーマ設定取得"""
        return self.theme_config.get(self.theme, self.theme_config['modern'])
        
    def create_interactive_timeseries(self, data: Dict[str, Any], 
                                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        インタラクティブ時系列チャート作成
        
        Args:
            data: 時系列データ（複数系列対応）
            config: チャート設定
            
        Returns:
            インタラクティブチャート設定
        """
        if config is None:
            config = {}
            
        theme = self._get_theme()
        
        if self.chart_library == 'plotly':
            return self._create_plotly_timeseries(data, config, theme)
        elif self.chart_library == 'd3':
            return self._create_d3_timeseries(data, config, theme)
        else:
            return self._create_plotly_timeseries(data, config, theme)  # フォールバック
            
    def _create_plotly_timeseries(self, data: Dict[str, Any], 
                                config: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Plotly.js時系列チャート作成"""
        
        # データ準備
        time_series = data.get('time_series', {})
        annotations = data.get('annotations', [])
        
        traces = []
        colors = [theme['primary_color'], theme['secondary_color'], theme['accent_color']]
        
        # 複数系列データ処理
        for i, (series_name, series_data) in enumerate(time_series.items()):
            timestamps = series_data.get('timestamps', [])
            values = series_data.get('values', [])
            
            # タイムスタンプ生成（未指定の場合）
            if not timestamps:
                timestamps = [f"2024-01-{i+1:02d}" for i in range(len(values))]
                
            trace = {
                'x': timestamps,
                'y': values,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': series_name,
                'line': {
                    'color': colors[i % len(colors)],
                    'width': 2
                },
                'marker': {
                    'size': 4,
                    'color': colors[i % len(colors)]
                },
                'hovertemplate': f'<b>{series_name}</b><br>' +
                               'Date: %{x}<br>' +
                               'Value: %{y:.2f}<br>' +
                               '<extra></extra>'
            }
            traces.append(trace)
            
        # 信頼区間があれば追加
        if 'confidence_bands' in data:
            conf_data = data['confidence_bands']
            timestamps = conf_data.get('timestamps', [])
            upper_bound = conf_data.get('upper_bound', [])
            lower_bound = conf_data.get('lower_bound', [])
            
            # 上限
            traces.append({
                'x': timestamps,
                'y': upper_bound,
                'type': 'scatter',
                'mode': 'lines',
                'name': '信頼区間上限',
                'line': {'color': 'rgba(0,0,0,0)'},
                'showlegend': False,
                'hoverinfo': 'skip'
            })
            
            # 下限（塗りつぶし）
            traces.append({
                'x': timestamps,
                'y': lower_bound,
                'type': 'scatter',
                'mode': 'lines',
                'name': '信頼区間',
                'fill': 'tonexty',
                'fillcolor': 'rgba(52, 152, 219, 0.2)',
                'line': {'color': 'rgba(52, 152, 219, 0.5)'},
                'hovertemplate': 'Date: %{x}<br>Lower: %{y:.2f}<extra></extra>'
            })
            
        # レイアウト設定
        layout = {
            'title': {
                'text': config.get('title', 'インタラクティブ時系列チャート'),
                'font': {
                    'size': 18,
                    'color': theme['text_color'],
                    'family': theme['font_family']
                }
            },
            'paper_bgcolor': theme['background'],
            'plot_bgcolor': theme['background'],
            'font': {
                'family': theme['font_family'],
                'size': theme['font_size'],
                'color': theme['text_color']
            },
            'xaxis': {
                'title': config.get('x_label', '時間'),
                'gridcolor': theme['grid_color'],
                'zeroline': False,
                'rangeslider': {'visible': True},  # レンジスライダー
                'rangeselector': {
                    'buttons': [
                        {'count': 7, 'label': '7日', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 30, 'label': '30日', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 90, 'label': '90日', 'step': 'day', 'stepmode': 'backward'},
                        {'step': 'all', 'label': '全期間'}
                    ]
                }
            },
            'yaxis': {
                'title': config.get('y_label', '値'),
                'gridcolor': theme['grid_color'],
                'zeroline': False
            },
            'hovermode': 'x unified',  # 統一ホバー
            'legend': {
                'orientation': 'h',
                'y': -0.2,
                'x': 0.5,
                'xanchor': 'center'
            },
            'margin': {'t': 60, 'r': 30, 'b': 80, 'l': 60}
        }
        
        # アノテーション追加
        if annotations:
            layout['annotations'] = []
            for ann in annotations:
                layout['annotations'].append({
                    'x': ann.get('x'),
                    'y': ann.get('y'),
                    'text': ann.get('text', ''),
                    'showarrow': True,
                    'arrowhead': 2,
                    'arrowcolor': theme['accent_color'],
                    'font': {'color': theme['text_color']}
                })
                
        plotly_config = {
            'data': traces,
            'layout': layout,
            'config': {
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'timeseries_chart',
                    'height': 600,
                    'width': 1000,
                    'scale': 2
                }
            }
        }
        
        return {
            'chart_config': plotly_config,
            'chart_library': 'plotly',
            'chart_id': f'interactive_ts_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'interactive_timeseries'
        }
        
    def _create_d3_timeseries(self, data: Dict[str, Any], 
                            config: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """D3.js時系列チャート作成"""
        
        # D3.js用データ構造
        time_series = data.get('time_series', {})
        d3_data = []
        
        for series_name, series_data in time_series.items():
            timestamps = series_data.get('timestamps', [])
            values = series_data.get('values', [])
            
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                d3_data.append({
                    'date': timestamp,
                    'value': value,
                    'series': series_name,
                    'index': i
                })
                
        # D3.js設定
        d3_config = {
            'data': d3_data,
            'config': {
                'width': config.get('width', 800),
                'height': config.get('height', 400),
                'margin': {'top': 20, 'right': 30, 'bottom': 40, 'left': 50},
                'theme': theme,
                'interactions': {
                    'zoom': True,
                    'pan': True,
                    'brush': True,
                    'tooltip': True
                },
                'animations': {
                    'duration': 750,
                    'easing': 'easeInOutQuart'
                }
            }
        }
        
        # D3.js JavaScript コード
        d3_js_code = f"""
        function createD3Timeseries(containerId, config) {{
            const data = config.data;
            const cfg = config.config;
            
            // SVG作成
            const svg = d3.select(`#${{containerId}}`)
                .append('svg')
                .attr('width', cfg.width)
                .attr('height', cfg.height);
                
            const g = svg.append('g')
                .attr('transform', `translate(${{cfg.margin.left}},${{cfg.margin.top}})`);
                
            const width = cfg.width - cfg.margin.left - cfg.margin.right;
            const height = cfg.height - cfg.margin.top - cfg.margin.bottom;
            
            // スケール設定
            const xScale = d3.scaleTime()
                .domain(d3.extent(data, d => new Date(d.date)))
                .range([0, width]);
                
            const yScale = d3.scaleLinear()
                .domain(d3.extent(data, d => d.value))
                .range([height, 0]);
                
            // ライン関数
            const line = d3.line()
                .x(d => xScale(new Date(d.date)))
                .y(d => yScale(d.value))
                .curve(d3.curveMonotoneX);
                
            // データ系列別グループ化
            const seriesData = d3.group(data, d => d.series);
            
            // 各系列のライン描画
            const colors = ['{theme['primary_color']}', '{theme['secondary_color']}', '{theme['accent_color']}'];
            let colorIndex = 0;
            
            seriesData.forEach((values, seriesName) => {{
                g.append('path')
                    .datum(values)
                    .attr('fill', 'none')
                    .attr('stroke', colors[colorIndex % colors.length])
                    .attr('stroke-width', 2)
                    .attr('d', line)
                    .attr('class', `line-${{seriesName.replace(/\\s+/g, '-')}}`);
                    
                colorIndex++;
            }});
            
            // 軸追加
            g.append('g')
                .attr('transform', `translate(0,${{height}})`)
                .call(d3.axisBottom(xScale));
                
            g.append('g')
                .call(d3.axisLeft(yScale));
                
            // ズーム機能
            if (cfg.interactions.zoom) {{
                const zoom = d3.zoom()
                    .scaleExtent([1, 10])
                    .on('zoom', function(event) {{
                        const newXScale = event.transform.rescaleX(xScale);
                        
                        // ライン更新
                        seriesData.forEach((values, seriesName) => {{
                            const newLine = d3.line()
                                .x(d => newXScale(new Date(d.date)))
                                .y(d => yScale(d.value))
                                .curve(d3.curveMonotoneX);
                                
                            g.select(`.line-${{seriesName.replace(/\\s+/g, '-')}}`)
                                .attr('d', newLine);
                        }});
                        
                        // X軸更新
                        g.select('.x-axis')
                            .call(d3.axisBottom(newXScale));
                    }});
                    
                svg.call(zoom);
            }}
            
            // ツールチップ
            if (cfg.interactions.tooltip) {{
                const tooltip = d3.select('body').append('div')
                    .attr('class', 'tooltip')
                    .style('opacity', 0)
                    .style('position', 'absolute')
                    .style('background', 'rgba(0,0,0,0.8)')
                    .style('color', 'white')
                    .style('padding', '8px')
                    .style('border-radius', '4px')
                    .style('pointer-events', 'none');
                    
                // ドット追加（ホバー用）
                seriesData.forEach((values, seriesName) => {{
                    g.selectAll(`.dot-${{seriesName.replace(/\\s+/g, '-')}}`)
                        .data(values)
                        .enter().append('circle')
                        .attr('class', `dot-${{seriesName.replace(/\\s+/g, '-')}}`)
                        .attr('cx', d => xScale(new Date(d.date)))
                        .attr('cy', d => yScale(d.value))
                        .attr('r', 3)
                        .style('opacity', 0)
                        .on('mouseover', function(event, d) {{
                            d3.select(this).style('opacity', 1);
                            tooltip.transition().duration(200).style('opacity', .9);
                            tooltip.html(`Series: ${{d.series}}<br/>Date: ${{d.date}}<br/>Value: ${{d.value.toFixed(2)}}`)
                                .style('left', (event.pageX + 10) + 'px')
                                .style('top', (event.pageY - 28) + 'px');
                        }})
                        .on('mouseout', function(d) {{
                            d3.select(this).style('opacity', 0);
                            tooltip.transition().duration(500).style('opacity', 0);
                        }});
                }});
            }}
        }}
        """
        
        return {
            'chart_config': d3_config,
            'chart_library': 'd3',
            'chart_id': f'interactive_d3_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'interactive_d3_timeseries',
            'js_code': d3_js_code
        }
        
    def create_multi_axis_chart(self, data: Dict[str, Any], 
                              config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        多軸チャート作成
        
        Args:
            data: 複数系列データ（異なる単位）
            config: チャート設定
            
        Returns:
            多軸チャート設定
        """
        if config is None:
            config = {}
            
        theme = self._get_theme()
        
        # データ系列と軸の対応
        left_axis_series = data.get('left_axis', {})
        right_axis_series = data.get('right_axis', {})
        
        traces = []
        colors = [theme['primary_color'], theme['secondary_color'], theme['accent_color']]
        color_index = 0
        
        # 左軸系列
        for series_name, series_data in left_axis_series.items():
            trace = {
                'x': series_data.get('timestamps', []),
                'y': series_data.get('values', []),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': series_name,
                'yaxis': 'y',
                'line': {'color': colors[color_index % len(colors)], 'width': 2},
                'marker': {'size': 4}
            }
            traces.append(trace)
            color_index += 1
            
        # 右軸系列
        for series_name, series_data in right_axis_series.items():
            trace = {
                'x': series_data.get('timestamps', []),
                'y': series_data.get('values', []),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': series_name,
                'yaxis': 'y2',
                'line': {'color': colors[color_index % len(colors)], 'width': 2, 'dash': 'dash'},
                'marker': {'size': 4}
            }
            traces.append(trace)
            color_index += 1
            
        layout = {
            'title': config.get('title', '多軸チャート'),
            'paper_bgcolor': theme['background'],
            'plot_bgcolor': theme['background'],
            'font': {'family': theme['font_family'], 'color': theme['text_color']},
            'xaxis': {
                'title': config.get('x_label', '時間'),
                'gridcolor': theme['grid_color']
            },
            'yaxis': {
                'title': config.get('left_y_label', '左軸'),
                'gridcolor': theme['grid_color'],
                'side': 'left'
            },
            'yaxis2': {
                'title': config.get('right_y_label', '右軸'),
                'gridcolor': theme['grid_color'],
                'side': 'right',
                'overlaying': 'y'
            },
            'legend': {'x': 0.1, 'y': 1.1, 'orientation': 'h'},
            'hovermode': 'x unified'
        }
        
        return {
            'chart_config': {'data': traces, 'layout': layout},
            'chart_library': 'plotly',
            'chart_id': f'multi_axis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'multi_axis'
        }
        
    def create_candlestick_chart(self, data: Dict[str, Any], 
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ローソク足チャート作成
        
        Args:
            data: OHLC（Open, High, Low, Close）データ
            config: チャート設定
            
        Returns:
            ローソク足チャート設定
        """
        if config is None:
            config = {}
            
        theme = self._get_theme()
        
        ohlc_data = data.get('ohlc', {})
        volume_data = data.get('volume', {})
        
        traces = []
        
        # ローソク足
        candlestick_trace = {
            'type': 'candlestick',
            'x': ohlc_data.get('timestamps', []),
            'open': ohlc_data.get('open', []),
            'high': ohlc_data.get('high', []),
            'low': ohlc_data.get('low', []),
            'close': ohlc_data.get('close', []),
            'name': 'OHLC',
            'increasing': {'line': {'color': '#00b050'}},
            'decreasing': {'line': {'color': '#ff0000'}},
            'yaxis': 'y'
        }
        traces.append(candlestick_trace)
        
        # 出来高（あれば）
        if volume_data:
            volume_trace = {
                'type': 'bar',
                'x': volume_data.get('timestamps', []),
                'y': volume_data.get('values', []),
                'name': '出来高',
                'yaxis': 'y2',
                'marker': {'color': 'rgba(52, 152, 219, 0.6)'}
            }
            traces.append(volume_trace)
            
        layout = {
            'title': config.get('title', 'ローソク足チャート'),
            'paper_bgcolor': theme['background'],
            'plot_bgcolor': theme['background'],
            'font': {'family': theme['font_family'], 'color': theme['text_color']},
            'xaxis': {
                'title': '時間',
                'gridcolor': theme['grid_color'],
                'rangeslider': {'visible': False}  # ローソク足では無効化
            },
            'yaxis': {
                'title': '価格',
                'gridcolor': theme['grid_color'],
                'domain': [0.3, 1] if volume_data else [0, 1]
            },
            'showlegend': True,
            'hovermode': 'x unified'
        }
        
        # 出来高用の2軸目
        if volume_data:
            layout['yaxis2'] = {
                'title': '出来高',
                'gridcolor': theme['grid_color'],
                'domain': [0, 0.25],
                'side': 'right'
            }
            
        return {
            'chart_config': {'data': traces, 'layout': layout},
            'chart_library': 'plotly',
            'chart_id': f'candlestick_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'candlestick'
        }
        
    def create_real_time_chart(self, initial_data: Dict[str, Any], 
                             config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        リアルタイムチャート作成
        
        Args:
            initial_data: 初期データ
            config: チャート設定（更新間隔含む）
            
        Returns:
            リアルタイムチャート設定
        """
        if config is None:
            config = {}
            
        theme = self._get_theme()
        update_interval = config.get('update_interval', 5000)  # 5秒
        
        # 基本チャート設定
        chart_config = self.create_interactive_timeseries(initial_data, config)
        
        # リアルタイム更新用JavaScript
        realtime_js = f"""
        function setupRealtimeChart(chartId, updateInterval) {{
            let chart = null;
            
            // チャート初期化
            if (typeof Plotly !== 'undefined') {{
                const container = document.getElementById(chartId);
                const config = {json.dumps(chart_config['chart_config'])};
                Plotly.newPlot(container, config.data, config.layout, config.config);
                chart = container;
            }}
            
            // データ更新関数
            function updateChartData() {{
                if (!chart) return;
                
                // 新しいデータポイント生成（ランダム例）
                const now = new Date();
                const newValue = Math.random() * 100 + 50;
                
                // データ追加
                const update = {{
                    x: [[now]],
                    y: [[newValue]]
                }};
                
                Plotly.extendTraces(chart, update, [0]);
                
                // 古いデータ削除（最新100ポイントのみ保持）
                if (chart.data[0].x.length > 100) {{
                    Plotly.relayout(chart, {{
                        'xaxis.range': [
                            chart.data[0].x[chart.data[0].x.length - 100],
                            chart.data[0].x[chart.data[0].x.length - 1]
                        ]
                    }});
                }}
                
                console.log('Chart updated at:', now);
            }}
            
            // 定期更新開始
            const intervalId = setInterval(updateChartData, updateInterval);
            
            // クリーンアップ関数
            return function cleanup() {{
                clearInterval(intervalId);
            }};
        }}
        """
        
        chart_config['realtime_js'] = realtime_js
        chart_config['update_interval'] = update_interval
        chart_config['chart_type'] = 'realtime'
        
        return chart_config
        
    def generate_interactive_html(self, charts: List[Dict[str, Any]], 
                                layout: str = 'tabs') -> str:
        """
        インタラクティブチャートHTML生成
        
        Args:
            charts: チャート設定リスト
            layout: レイアウト ('tabs', 'grid', 'stacked')
            
        Returns:
            HTML文字列
        """
        theme = self._get_theme()
        
        # CSS
        css_styles = f"""
        <style>
            body {{
                font-family: {theme['font_family']};
                background-color: {theme['background']};
                color: {theme['text_color']};
                margin: 0;
                padding: 20px;
            }}
            
            .chart-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }}
            
            .tabs {{
                border-bottom: 2px solid {theme['grid_color']};
                margin-bottom: 20px;
            }}
            
            .tab-button {{
                background: none;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                color: {theme['text_color']};
            }}
            
            .tab-button.active {{
                border-bottom-color: {theme['primary_color']};
                color: {theme['primary_color']};
                font-weight: bold;
            }}
            
            .tab-content {{
                display: none;
                min-height: 500px;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }}
            
            .chart-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: {theme['text_color']};
            }}
            
            .loading {{
                text-align: center;
                padding: 50px;
                color: {theme['text_color']};
            }}
        </style>
        """
        
        # タブ用JavaScript
        tab_js = """
        <script>
            function showTab(tabId) {
                // 全タブ非表示
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // 全ボタン非アクティブ
                const buttons = document.querySelectorAll('.tab-button');
                buttons.forEach(button => button.classList.remove('active'));
                
                // 選択タブ表示
                document.getElementById(tabId).classList.add('active');
                document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
            }
            
            // 初期化
            document.addEventListener('DOMContentLoaded', function() {
                const firstTab = document.querySelector('.tab-content');
                if (firstTab) {
                    showTab(firstTab.id);
                }
            });
        </script>
        """
        
        # チャートJavaScript
        chart_scripts = []
        for chart in charts:
            chart_id = chart['chart_id']
            
            if chart['chart_library'] == 'plotly':
                config = json.dumps(chart['chart_config'])
                script = f"""
                Plotly.newPlot('{chart_id}', {config}.data, {config}.layout, {config}.config);
                """
                chart_scripts.append(script)
                
            elif chart['chart_library'] == 'd3':
                if 'js_code' in chart:
                    script = f"""
                    {chart['js_code']}
                    createD3Timeseries('{chart_id}', {json.dumps(chart['chart_config'])});
                    """
                    chart_scripts.append(script)
                    
            # リアルタイムチャート
            if chart.get('chart_type') == 'realtime' and 'realtime_js' in chart:
                script = f"""
                {chart['realtime_js']}
                setupRealtimeChart('{chart_id}', {chart.get('update_interval', 5000)});
                """
                chart_scripts.append(script)
                
        # HTMLレイアウト生成
        if layout == 'tabs':
            html_body = self._generate_tabs_layout(charts, theme)
        elif layout == 'grid':
            html_body = self._generate_grid_layout(charts, theme)
        else:
            html_body = self._generate_stacked_layout(charts, theme)
            
        # 完整HTML
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>インタラクティブチャート</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            {css_styles}
        </head>
        <body>
            {html_body}
            
            {tab_js}
            
            <script>
                {' '.join(chart_scripts)}
            </script>
        </body>
        </html>
        """
        
        return html_template
        
    def _generate_tabs_layout(self, charts: List[Dict[str, Any]], 
                            theme: Dict[str, Any]) -> str:
        """タブレイアウト生成"""
        tabs_html = '<div class="tabs">'
        content_html = ''
        
        for i, chart in enumerate(charts):
            chart_id = chart['chart_id']
            chart_title = chart.get('title', f'チャート {i+1}')
            
            # タブボタン
            tabs_html += f'<button class="tab-button" onclick="showTab(\'{chart_id}_content\')">{chart_title}</button>'
            
            # タブコンテンツ
            content_html += f"""
            <div id="{chart_id}_content" class="tab-content">
                <div class="chart-container">
                    <div class="chart-title">{chart_title}</div>
                    <div id="{chart_id}" style="width:100%;height:500px;"></div>
                </div>
            </div>
            """
            
        tabs_html += '</div>'
        
        return tabs_html + content_html
        
    def _generate_grid_layout(self, charts: List[Dict[str, Any]], 
                            theme: Dict[str, Any]) -> str:
        """グリッドレイアウト生成"""
        grid_html = '<div class="chart-grid">'
        
        for chart in charts:
            chart_id = chart['chart_id']
            chart_title = chart.get('title', 'チャート')
            
            grid_html += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_title}</div>
                <div id="{chart_id}" style="width:100%;height:400px;"></div>
            </div>
            """
            
        grid_html += '</div>'
        return grid_html
        
    def _generate_stacked_layout(self, charts: List[Dict[str, Any]], 
                               theme: Dict[str, Any]) -> str:
        """スタックレイアウト生成"""
        stacked_html = ''
        
        for chart in charts:
            chart_id = chart['chart_id']
            chart_title = chart.get('title', 'チャート')
            
            stacked_html += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_title}</div>
                <div id="{chart_id}" style="width:100%;height:500px;"></div>
            </div>
            """
            
        return stacked_html
        
    def get_builder_config(self) -> Dict[str, Any]:
        """ビルダー設定取得"""
        return {
            'chart_library': self.chart_library,
            'theme': self.theme,
            'supported_libraries': ['plotly', 'd3', 'highcharts'],
            'supported_chart_types': [
                'interactive_timeseries',
                'multi_axis',
                'candlestick',
                'realtime'
            ],
            'layout_options': ['tabs', 'grid', 'stacked'],
            'theme_options': list(self.theme_config.keys())
        }