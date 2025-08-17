"""
3Dリスクマップ・ヒートマップ生成

相関マトリックス、リスクマップ、セクター分析、
地理的リスク分布、時間変化を3D・ヒートマップで可視化。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime, timedelta
import logging
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class RiskHeatmapGenerator:
    """
    3Dリスクマップ・ヒートマップジェネレーター
    
    相関ヒートマップ、リスクマトリックス、セクター分析、
    地理的分布、時間進化の3D可視化を提供。
    """
    
    def __init__(self, viz_style: str = '3d_modern', color_intensity: str = 'high'):
        """
        Args:
            viz_style: 可視化スタイル ('3d_modern', 'heatmap_classic', 'interactive')
            color_intensity: 色強度 ('low', 'medium', 'high')
        """
        self.viz_style = viz_style
        self.color_intensity = color_intensity
        self.color_schemes = self._init_color_schemes()
        
    def _init_color_schemes(self) -> Dict[str, Dict[str, Any]]:
        """カラースキーム初期化"""
        base_schemes = {
            'risk_gradient': {
                'low': '#2ecc71',      # 緑
                'medium': '#f39c12',   # オレンジ  
                'high': '#e74c3c',     # 赤
                'extreme': '#8e44ad'   # 紫
            },
            'correlation': {
                'negative': '#3498db', # 青（負の相関）
                'neutral': '#ecf0f1',  # グレー（無相関）
                'positive': '#e74c3c'  # 赤（正の相関）
            },
            'sector': {
                'technology': '#3498db',
                'finance': '#2ecc71',
                'healthcare': '#9b59b6',
                'energy': '#f39c12',
                'consumer': '#e74c3c',
                'industrial': '#95a5a6',
                'utilities': '#1abc9c',
                'materials': '#34495e'
            }
        }
        
        # 強度調整
        intensity_multipliers = {
            'low': 0.6,
            'medium': 0.8,
            'high': 1.0
        }
        
        multiplier = intensity_multipliers.get(self.color_intensity, 1.0)
        
        # 透明度調整
        for scheme_name, scheme in base_schemes.items():
            for key, color in scheme.items():
                if color.startswith('#'):
                    # RGB変換して透明度調整
                    base_schemes[scheme_name][key] = color
                    
        return base_schemes
        
    def create_correlation_heatmap(self, correlation_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        相関ヒートマップ作成
        
        Args:
            correlation_data: 相関マトリックスデータ
            config: 設定
            
        Returns:
            相関ヒートマップ設定
        """
        if config is None:
            config = {}
            
        colors = self.color_schemes['correlation']
        
        # 相関マトリックス準備
        correlation_matrix = correlation_data.get('correlation_matrix', {})
        assets = list(correlation_matrix.keys())
        
        # 2D相関マトリックス作成
        matrix_2d = []
        for asset1 in assets:
            row = []
            for asset2 in assets:
                corr_value = correlation_matrix.get(asset1, {}).get(asset2, 0)
                row.append(corr_value)
            matrix_2d.append(row)
            
        # クラスタリング（階層的）
        if len(assets) > 2:
            # 距離マトリックス作成（1-相関係数）
            distance_matrix = 1 - np.abs(np.array(matrix_2d))
            
            # 対称行列の上三角部分を使用
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # 階層クラスタリング
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # デンドログラム順序取得
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # 順序並び替え
            reordered_assets = [assets[i] for i in cluster_order]
            reordered_matrix = []
            for i in cluster_order:
                reordered_row = [matrix_2d[i][j] for j in cluster_order]
                reordered_matrix.append(reordered_row)
        else:
            reordered_assets = assets
            reordered_matrix = matrix_2d
            
        # ヒートマップチャート設定
        heatmap_config = {
            'data': [{
                'z': reordered_matrix,
                'x': reordered_assets,
                'y': reordered_assets,
                'type': 'heatmap',
                'colorscale': [
                    [0, colors['negative']],
                    [0.5, colors['neutral']],
                    [1, colors['positive']]
                ],
                'zmid': 0,
                'colorbar': {
                    'title': '相関係数',
                    'titleside': 'right'
                },
                'hovertemplate': '%{y} vs %{x}<br>相関: %{z:.3f}<extra></extra>',
                'showscale': True
            }],
            'layout': {
                'title': {
                    'text': config.get('title', '資産相関ヒートマップ'),
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': '資産',
                    'tickangle': 45
                },
                'yaxis': {
                    'title': '資産'
                },
                'width': config.get('width', 600),
                'height': config.get('height', 600),
                'margin': {'l': 100, 'r': 100, 't': 100, 'b': 100}
            }
        }
        
        # 3D表面プロット版
        surface_config = {
            'data': [{
                'z': reordered_matrix,
                'x': reordered_assets,
                'y': reordered_assets,
                'type': 'surface',
                'colorscale': [
                    [0, colors['negative']],
                    [0.5, colors['neutral']],
                    [1, colors['positive']]
                ],
                'showscale': True,
                'hovertemplate': '%{y} vs %{x}<br>相関: %{z:.3f}<extra></extra>'
            }],
            'layout': {
                'title': config.get('title', '3D相関表面'),
                'scene': {
                    'xaxis': {'title': '資産X'},
                    'yaxis': {'title': '資産Y'},
                    'zaxis': {'title': '相関係数'},
                    'camera': {
                        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    }
                }
            }
        }
        
        return {
            'heatmap_2d': heatmap_config,
            'surface_3d': surface_config,
            'reordered_assets': reordered_assets,
            'cluster_info': {
                'method': 'hierarchical_ward',
                'reordering_applied': len(assets) > 2
            },
            'chart_id': f'correlation_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'correlation_heatmap'
        }
        
    def create_risk_matrix_heatmap(self, risk_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        リスクマトリックスヒートマップ作成
        
        Args:
            risk_data: リスクデータ（確率×影響度）
            config: 設定
            
        Returns:
            リスクマトリックス設定
        """
        if config is None:
            config = {}
            
        colors = self.color_schemes['risk_gradient']
        
        # リスクデータ整理
        risks = risk_data.get('risks', [])
        
        # リスクマトリックス軸設定
        probability_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        impact_levels = ['Minimal', 'Minor', 'Moderate', 'Major', 'Severe']
        
        # マトリックス初期化
        risk_matrix = np.zeros((len(impact_levels), len(probability_levels)))
        risk_details = {}
        
        # リスクデータをマトリックスに配置
        for risk in risks:
            prob_idx = risk.get('probability_index', 2)  # 0-4
            impact_idx = risk.get('impact_index', 2)     # 0-4
            
            # 有効範囲チェック
            if 0 <= prob_idx < len(probability_levels) and 0 <= impact_idx < len(impact_levels):
                risk_score = prob_idx * impact_idx  # リスクスコア
                risk_matrix[impact_idx, prob_idx] += risk_score
                
                # 詳細情報保存
                cell_key = f"{impact_idx}_{prob_idx}"
                if cell_key not in risk_details:
                    risk_details[cell_key] = []
                risk_details[cell_key].append({
                    'name': risk.get('name', 'Unknown Risk'),
                    'score': risk_score,
                    'description': risk.get('description', '')
                })
                
        # カスタムホバーテキスト作成
        hover_text = []
        for i in range(len(impact_levels)):
            hover_row = []
            for j in range(len(probability_levels)):
                cell_key = f"{i}_{j}"
                if cell_key in risk_details:
                    risks_in_cell = risk_details[cell_key]
                    hover_info = f"影響度: {impact_levels[i]}<br>確率: {probability_levels[j]}<br>"
                    hover_info += f"リスク数: {len(risks_in_cell)}<br>"
                    hover_info += "\\n".join([f"• {r['name']}" for r in risks_in_cell[:3]])
                    if len(risks_in_cell) > 3:
                        hover_info += f"\\n... (+{len(risks_in_cell) - 3})"
                else:
                    hover_info = f"影響度: {impact_levels[i]}<br>確率: {probability_levels[j]}<br>リスク数: 0"
                hover_row.append(hover_info)
            hover_text.append(hover_row)
            
        # リスクレベル色分け
        risk_colorscale = [
            [0.0, colors['low']],
            [0.3, colors['medium']],
            [0.7, colors['high']],
            [1.0, colors['extreme']]
        ]
        
        heatmap_config = {
            'data': [{
                'z': risk_matrix.tolist(),
                'x': probability_levels,
                'y': impact_levels,
                'type': 'heatmap',
                'colorscale': risk_colorscale,
                'hovertemplate': '%{hovertext}<extra></extra>',
                'hovertext': hover_text,
                'colorbar': {
                    'title': 'リスクスコア',
                    'titleside': 'right'
                }
            }],
            'layout': {
                'title': {
                    'text': config.get('title', 'リスクマトリックス'),
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': '発生確率',
                    'tickangle': 0
                },
                'yaxis': {
                    'title': '影響度'
                },
                'annotations': self._create_risk_matrix_annotations(risk_matrix, probability_levels, impact_levels),
                'width': config.get('width', 700),
                'height': config.get('height', 500)
            }
        }
        
        return {
            'heatmap_config': heatmap_config,
            'risk_details': risk_details,
            'risk_summary': {
                'total_risks': len(risks),
                'high_risk_count': np.sum(risk_matrix > 12),  # 高リスク閾値
                'max_risk_score': np.max(risk_matrix),
                'avg_risk_score': np.mean(risk_matrix[risk_matrix > 0])
            },
            'chart_id': f'risk_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'risk_matrix'
        }
        
    def create_sector_risk_heatmap(self, sector_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        セクター別リスクヒートマップ作成
        
        Args:
            sector_data: セクター別リスクデータ
            config: 設定
            
        Returns:
            セクターヒートマップ設定
        """
        if config is None:
            config = {}
            
        sector_colors = self.color_schemes['sector']
        
        # セクターデータ整理
        sectors = list(sector_data.keys())
        risk_metrics = config.get('risk_metrics', ['volatility', 'var_95', 'max_drawdown', 'correlation'])
        
        # リスクマトリックス作成
        risk_matrix = []
        for sector in sectors:
            sector_risks = sector_data[sector]
            row = []
            for metric in risk_metrics:
                value = sector_risks.get(metric, 0)
                row.append(value)
            risk_matrix.append(row)
            
        # 正規化（0-1スケール）
        risk_array = np.array(risk_matrix)
        normalized_matrix = []
        for col in range(len(risk_metrics)):
            column_values = risk_array[:, col]
            if np.max(column_values) > np.min(column_values):
                normalized_col = (column_values - np.min(column_values)) / (np.max(column_values) - np.min(column_values))
            else:
                normalized_col = np.zeros_like(column_values)
            normalized_matrix.append(normalized_col)
            
        normalized_matrix = np.array(normalized_matrix).T
        
        # ヒートマップ設定
        heatmap_config = {
            'data': [{
                'z': normalized_matrix.tolist(),
                'x': risk_metrics,
                'y': sectors,
                'type': 'heatmap',
                'colorscale': [
                    [0, '#2ecc71'],    # 低リスク（緑）
                    [0.5, '#f39c12'],  # 中リスク（オレンジ）
                    [1, '#e74c3c']     # 高リスク（赤）
                ],
                'hovertemplate': 'セクター: %{y}<br>指標: %{x}<br>リスクレベル: %{z:.3f}<extra></extra>',
                'colorbar': {
                    'title': '正規化リスク<br>(0=低, 1=高)',
                    'titleside': 'right'
                }
            }],
            'layout': {
                'title': {
                    'text': config.get('title', 'セクター別リスクヒートマップ'),
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': 'リスク指標',
                    'tickangle': 45
                },
                'yaxis': {
                    'title': 'セクター'
                },
                'width': config.get('width', 800),
                'height': config.get('height', 600)
            }
        }
        
        # セクター別サマリー
        sector_summary = {}
        for i, sector in enumerate(sectors):
            sector_risk_avg = np.mean(normalized_matrix[i])
            risk_level = 'Low' if sector_risk_avg < 0.3 else 'Medium' if sector_risk_avg < 0.7 else 'High'
            
            sector_summary[sector] = {
                'avg_risk_score': sector_risk_avg,
                'risk_level': risk_level,
                'highest_risk_metric': risk_metrics[np.argmax(normalized_matrix[i])],
                'lowest_risk_metric': risk_metrics[np.argmin(normalized_matrix[i])]
            }
            
        return {
            'heatmap_config': heatmap_config,
            'sector_summary': sector_summary,
            'original_data': risk_matrix,
            'normalized_data': normalized_matrix.tolist(),
            'chart_id': f'sector_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'sector_risk_heatmap'
        }
        
    def create_temporal_risk_heatmap(self, temporal_data: Dict[str, Any],
                                   config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        時間変化リスクヒートマップ作成
        
        Args:
            temporal_data: 時系列リスクデータ
            config: 設定
            
        Returns:
            時間変化ヒートマップ設定
        """
        if config is None:
            config = {}
            
        # 時系列データ整理
        time_periods = temporal_data.get('time_periods', [])
        assets = temporal_data.get('assets', [])
        risk_values = temporal_data.get('risk_matrix', [])  # [time][asset]
        
        # 時間軸ラベル
        if not time_periods:
            time_periods = [f'T{i+1}' for i in range(len(risk_values))]
            
        # ヒートマップ設定
        heatmap_config = {
            'data': [{
                'z': risk_values,
                'x': assets,
                'y': time_periods,
                'type': 'heatmap',
                'colorscale': [
                    [0, '#2ecc71'],
                    [0.25, '#f1c40f'],
                    [0.5, '#e67e22'],
                    [0.75, '#e74c3c'],
                    [1, '#8e44ad']
                ],
                'hovertemplate': '時期: %{y}<br>資産: %{x}<br>リスク値: %{z:.3f}<extra></extra>',
                'colorbar': {
                    'title': 'リスク値',
                    'titleside': 'right'
                }
            }],
            'layout': {
                'title': {
                    'text': config.get('title', '時間変化リスクヒートマップ'),
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': '資産',
                    'tickangle': 45
                },
                'yaxis': {
                    'title': '時期'
                },
                'width': config.get('width', 900),
                'height': config.get('height', 600)
            }
        }
        
        # 3Dサーフェス版
        # メッシュグリッド作成
        X, Y = np.meshgrid(range(len(assets)), range(len(time_periods)))
        Z = np.array(risk_values)
        
        surface_3d_config = {
            'data': [{
                'x': X.tolist(),
                'y': Y.tolist(),
                'z': Z.tolist(),
                'type': 'surface',
                'colorscale': [
                    [0, '#2ecc71'],
                    [0.5, '#f39c12'],
                    [1, '#e74c3c']
                ],
                'hovertemplate': 'Asset: %{x}<br>Time: %{y}<br>Risk: %{z:.3f}<extra></extra>'
            }],
            'layout': {
                'title': config.get('title', '3D時間変化リスク表面'),
                'scene': {
                    'xaxis': {
                        'title': '資産インデックス',
                        'tickvals': list(range(len(assets))),
                        'ticktext': assets
                    },
                    'yaxis': {
                        'title': '時期インデックス', 
                        'tickvals': list(range(len(time_periods))),
                        'ticktext': time_periods
                    },
                    'zaxis': {'title': 'リスク値'},
                    'camera': {
                        'eye': {'x': 1.2, 'y': 1.2, 'z': 1.2}
                    }
                }
            }
        }
        
        # 時系列統計
        temporal_stats = {
            'max_risk_period': time_periods[np.unravel_index(np.argmax(Z), Z.shape)[0]],
            'max_risk_asset': assets[np.unravel_index(np.argmax(Z), Z.shape)[1]],
            'avg_risk_by_period': [np.mean(period_risks) for period_risks in risk_values],
            'avg_risk_by_asset': [np.mean([risk_values[t][a] for t in range(len(time_periods))]) 
                                for a in range(len(assets))],
            'risk_trend': 'increasing' if np.corrcoef(range(len(time_periods)), 
                                                   [np.mean(risks) for risks in risk_values])[0,1] > 0 else 'decreasing'
        }
        
        return {
            'heatmap_2d': heatmap_config,
            'surface_3d': surface_3d_config,
            'temporal_stats': temporal_stats,
            'chart_id': f'temporal_risk_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'temporal_risk_heatmap'
        }
        
    def create_geographic_risk_map(self, geographic_data: Dict[str, Any],
                                 config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        地理的リスクマップ作成
        
        Args:
            geographic_data: 地理的リスクデータ
            config: 設定
            
        Returns:
            地理的リスクマップ設定
        """
        if config is None:
            config = {}
            
        # 地理データ整理
        countries = geographic_data.get('countries', [])
        risk_scores = geographic_data.get('risk_scores', [])
        country_codes = geographic_data.get('country_codes', [])
        
        # 地理的ヒートマップ（コロプレス）
        choropleth_config = {
            'data': [{
                'type': 'choropleth',
                'locations': country_codes,
                'z': risk_scores,
                'text': countries,
                'colorscale': [
                    [0, '#2ecc71'],
                    [0.5, '#f39c12'],
                    [1, '#e74c3c']
                ],
                'hovertemplate': '国: %{text}<br>リスクスコア: %{z:.2f}<extra></extra>',
                'colorbar': {
                    'title': 'リスクスコア',
                    'titleside': 'right'
                }
            }],
            'layout': {
                'title': {
                    'text': config.get('title', '地理的リスク分布'),
                    'font': {'size': 16}
                },
                'geo': {
                    'showframe': False,
                    'showcoastlines': True,
                    'projection': {'type': 'equirectangular'}
                },
                'width': config.get('width', 1000),
                'height': config.get('height', 600)
            }
        }
        
        # 散布図マップ版（都市レベル）
        if 'cities' in geographic_data:
            cities_data = geographic_data['cities']
            
            scatter_map_config = {
                'data': [{
                    'type': 'scattergeo',
                    'lon': [city['longitude'] for city in cities_data],
                    'lat': [city['latitude'] for city in cities_data],
                    'text': [city['name'] for city in cities_data],
                    'marker': {
                        'size': [city['risk_score'] * 10 for city in cities_data],
                        'color': [city['risk_score'] for city in cities_data],
                        'colorscale': [
                            [0, '#2ecc71'],
                            [0.5, '#f39c12'],
                            [1, '#e74c3c']
                        ],
                        'sizemode': 'diameter',
                        'colorbar': {
                            'title': 'リスクスコア'
                        }
                    },
                    'hovertemplate': '都市: %{text}<br>リスク: %{marker.color:.2f}<extra></extra>'
                }],
                'layout': {
                    'title': config.get('title', '都市別リスクマップ'),
                    'geo': {
                        'projection': {'type': 'natural earth'},
                        'showland': True,
                        'landcolor': 'rgb(243, 243, 243)',
                        'coastlinecolor': 'rgb(204, 204, 204)'
                    }
                }
            }
        else:
            scatter_map_config = None
            
        # 地理統計
        geo_stats = {
            'highest_risk_country': countries[np.argmax(risk_scores)] if countries and risk_scores else 'N/A',
            'lowest_risk_country': countries[np.argmin(risk_scores)] if countries and risk_scores else 'N/A',
            'avg_global_risk': np.mean(risk_scores) if risk_scores else 0,
            'risk_variance': np.var(risk_scores) if risk_scores else 0,
            'high_risk_count': sum(1 for score in risk_scores if score > 0.7) if risk_scores else 0
        }
        
        result = {
            'choropleth_map': choropleth_config,
            'geo_stats': geo_stats,
            'chart_id': f'geo_risk_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'chart_type': 'geographic_risk_map'
        }
        
        if scatter_map_config:
            result['scatter_map'] = scatter_map_config
            
        return result
        
    def _create_risk_matrix_annotations(self, risk_matrix: np.ndarray,
                                      prob_levels: List[str], 
                                      impact_levels: List[str]) -> List[Dict[str, Any]]:
        """リスクマトリックス注釈作成"""
        annotations = []
        
        for i in range(len(impact_levels)):
            for j in range(len(prob_levels)):
                value = risk_matrix[i, j]
                if value > 0:
                    annotations.append({
                        'x': prob_levels[j],
                        'y': impact_levels[i],
                        'text': f'{value:.1f}',
                        'showarrow': False,
                        'font': {
                            'color': 'white' if value > np.max(risk_matrix) * 0.5 else 'black',
                            'size': 12,
                            'family': 'Arial, sans-serif'
                        }
                    })
                    
        return annotations
        
    def generate_comprehensive_heatmap_report(self, all_risk_data: Dict[str, Any],
                                            config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        包括的ヒートマップレポート生成
        
        Args:
            all_risk_data: 全リスクデータ
            config: レポート設定
            
        Returns:
            包括的レポート
        """
        if config is None:
            config = {}
            
        heatmaps = {}
        
        # 相関ヒートマップ
        if 'correlation_data' in all_risk_data:
            heatmaps['correlation'] = self.create_correlation_heatmap(
                all_risk_data['correlation_data'],
                {'title': '資産相関分析'}
            )
            
        # リスクマトリックス
        if 'risk_matrix_data' in all_risk_data:
            heatmaps['risk_matrix'] = self.create_risk_matrix_heatmap(
                all_risk_data['risk_matrix_data'],
                {'title': 'リスクマトリックス分析'}
            )
            
        # セクター分析
        if 'sector_data' in all_risk_data:
            heatmaps['sector'] = self.create_sector_risk_heatmap(
                all_risk_data['sector_data'],
                {'title': 'セクター別リスク分析'}
            )
            
        # 時間変化
        if 'temporal_data' in all_risk_data:
            heatmaps['temporal'] = self.create_temporal_risk_heatmap(
                all_risk_data['temporal_data'],
                {'title': '時間変化リスク分析'}
            )
            
        # 地理的分布
        if 'geographic_data' in all_risk_data:
            heatmaps['geographic'] = self.create_geographic_risk_map(
                all_risk_data['geographic_data'],
                {'title': '地理的リスク分布'}
            )
            
        return {
            'heatmaps': heatmaps,
            'report_id': f'heatmap_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'report_type': 'comprehensive_heatmap',
            'generation_time': datetime.now().isoformat(),
            'viz_style': self.viz_style,
            'color_intensity': self.color_intensity
        }
        
    def generate_heatmap_html(self, heatmap_report: Dict[str, Any]) -> str:
        """
        ヒートマップHTMLレポート生成
        
        Args:
            heatmap_report: ヒートマップレポートデータ
            
        Returns:
            HTML文字列
        """
        heatmaps = heatmap_report.get('heatmaps', {})
        
        # CSS
        css_styles = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            
            .report-container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .report-header {
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .heatmap-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 30px;
                padding: 30px;
            }
            
            .heatmap-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            
            .heatmap-card:hover {
                transform: translateY(-5px);
            }
            
            .heatmap-title {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 20px;
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            
            .chart-container {
                height: 500px;
                margin: 20px 0;
            }
            
            .stats-summary {
                background: #ecf0f1;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
            }
            
            .stat-item {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 5px 0;
                border-bottom: 1px solid #bdc3c7;
            }
            
            .stat-value {
                font-weight: bold;
                color: #e74c3c;
            }
        </style>
        """
        
        # チャートHTML生成
        chart_htmls = []
        chart_scripts = []
        
        for heatmap_name, heatmap_data in heatmaps.items():
            chart_id = heatmap_data.get('chart_id', f'heatmap_{heatmap_name}')
            
            if heatmap_name == 'correlation':
                chart_html = f"""
                <div class="heatmap-card">
                    <div class="heatmap-title">相関分析ヒートマップ</div>
                    <div class="chart-container" id="{chart_id}"></div>
                    <div class="stats-summary">
                        <div class="stat-item">
                            <span>クラスタリング適用:</span>
                            <span class="stat-value">{"はい" if heatmap_data.get('cluster_info', {}).get('reordering_applied') else "いいえ"}</span>
                        </div>
                    </div>
                </div>
                """
                chart_htmls.append(chart_html)
                
                config = json.dumps(heatmap_data['heatmap_2d'])
                chart_scripts.append(f"Plotly.newPlot('{chart_id}', {config}.data, {config}.layout);")
                
            elif heatmap_name == 'risk_matrix':
                chart_html = f"""
                <div class="heatmap-card">
                    <div class="heatmap-title">リスクマトリックス</div>
                    <div class="chart-container" id="{chart_id}"></div>
                    <div class="stats-summary">
                        {self._generate_risk_matrix_stats(heatmap_data.get('risk_summary', {}))}
                    </div>
                </div>
                """
                chart_htmls.append(chart_html)
                
                config = json.dumps(heatmap_data['heatmap_config'])
                chart_scripts.append(f"Plotly.newPlot('{chart_id}', {config}.data, {config}.layout);")
                
            elif heatmap_name == 'sector':
                chart_html = f"""
                <div class="heatmap-card">
                    <div class="heatmap-title">セクター別リスク分析</div>
                    <div class="chart-container" id="{chart_id}"></div>
                    <div class="stats-summary">
                        {self._generate_sector_stats(heatmap_data.get('sector_summary', {}))}
                    </div>
                </div>
                """
                chart_htmls.append(chart_html)
                
                config = json.dumps(heatmap_data['heatmap_config'])
                chart_scripts.append(f"Plotly.newPlot('{chart_id}', {config}.data, {config}.layout);")
                
        # HTML構築
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>リスクヒートマップ分析レポート</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {css_styles}
        </head>
        <body>
            <div class="report-container">
                <div class="report-header">
                    <h1>リスクヒートマップ分析レポート</h1>
                    <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                    <p>可視化スタイル: {heatmap_report.get('viz_style', 'N/A')} | 色強度: {heatmap_report.get('color_intensity', 'N/A')}</p>
                </div>
                
                <div class="heatmap-grid">
                    {''.join(chart_htmls)}
                </div>
            </div>
            
            <script>
                {' '.join(chart_scripts)}
            </script>
        </body>
        </html>
        """
        
        return html_content
        
    def _generate_risk_matrix_stats(self, risk_summary: Dict[str, Any]) -> str:
        """リスクマトリックス統計HTML生成"""
        stats_html = []
        for key, value in risk_summary.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                stats_html.append(f"""
                <div class="stat-item">
                    <span>{key.replace('_', ' ').title()}:</span>
                    <span class="stat-value">{formatted_value}</span>
                </div>
                """)
        return ''.join(stats_html)
        
    def _generate_sector_stats(self, sector_summary: Dict[str, Any]) -> str:
        """セクター統計HTML生成"""
        stats_html = []
        for sector, stats in sector_summary.items():
            risk_level = stats.get('risk_level', 'Unknown')
            avg_risk = stats.get('avg_risk_score', 0)
            stats_html.append(f"""
            <div class="stat-item">
                <span>{sector}:</span>
                <span class="stat-value">{risk_level} ({avg_risk:.3f})</span>
            </div>
            """)
        return ''.join(stats_html)
        
    def get_generator_config(self) -> Dict[str, Any]:
        """ジェネレーター設定取得"""
        return {
            'viz_style': self.viz_style,
            'color_intensity': self.color_intensity,
            'supported_heatmap_types': [
                'correlation_heatmap',
                'risk_matrix',
                'sector_risk_heatmap',
                'temporal_risk_heatmap',
                'geographic_risk_map'
            ],
            'available_styles': ['3d_modern', 'heatmap_classic', 'interactive'],
            'color_intensities': ['low', 'medium', 'high'],
            'color_schemes': list(self.color_schemes.keys())
        }