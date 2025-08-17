"""
シナリオ分析・モンテカルロシミュレーション

複数の市場シナリオでの予測分析、リスク評価、不確実性定量化を提供。
モンテカルロシミュレーション、VaR計算、ストレステスト、
感度分析を含む包括的なリスク分析システム。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from scipy import stats
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ScenarioAnalyzer:
    """
    シナリオ分析・モンテカルロシミュレーション
    
    複数シナリオでの予測、リスク指標計算、不確実性定量化、
    ストレステスト、感度分析を提供。
    """
    
    def __init__(self, n_simulations: int = 1000, confidence_levels: List[float] = None,
                 random_seed: int = 42):
        """
        Args:
            n_simulations: モンテカルロシミュレーション回数
            confidence_levels: 信頼水準リスト（VaR計算用）
            random_seed: 乱数シード
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]
        self.random_seed = random_seed
        self.simulation_results = {}
        np.random.seed(random_seed)
        
    def _generate_scenarios(self, base_data: np.ndarray, 
                          scenario_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """シナリオデータ生成"""
        scenarios = {}
        
        # ベースライン統計
        base_mean = np.mean(base_data)
        base_std = np.std(base_data)
        base_returns = np.diff(base_data) / base_data[:-1] if len(base_data) > 1 else np.array([0])
        
        # ベースラインシナリオ
        scenarios['baseline'] = base_data.copy()
        
        # 楽観シナリオ（高成長）
        optimistic_trend = scenario_params.get('optimistic_trend', 1.2)
        optimistic_volatility = scenario_params.get('optimistic_volatility', 0.8)
        optimistic_data = base_data * optimistic_trend
        optimistic_noise = np.random.normal(0, base_std * optimistic_volatility, len(base_data))
        scenarios['optimistic'] = optimistic_data + optimistic_noise
        
        # 悲観シナリオ（低成長・高ボラティリティ）
        pessimistic_trend = scenario_params.get('pessimistic_trend', 0.8)
        pessimistic_volatility = scenario_params.get('pessimistic_volatility', 1.5)
        pessimistic_data = base_data * pessimistic_trend
        pessimistic_noise = np.random.normal(0, base_std * pessimistic_volatility, len(base_data))
        scenarios['pessimistic'] = pessimistic_data + pessimistic_noise
        
        # ストレスシナリオ（極端な市場環境）
        stress_scenarios = scenario_params.get('stress_scenarios', {})
        
        # 金融危機シナリオ
        if 'financial_crisis' in stress_scenarios:
            crisis_params = stress_scenarios['financial_crisis']
            crisis_drop = crisis_params.get('drop_factor', 0.6)
            crisis_volatility = crisis_params.get('volatility_factor', 2.0)
            crisis_data = base_data * crisis_drop
            crisis_noise = np.random.normal(0, base_std * crisis_volatility, len(base_data))
            scenarios['financial_crisis'] = crisis_data + crisis_noise
            
        # インフレ高進シナリオ
        if 'high_inflation' in stress_scenarios:
            inflation_params = stress_scenarios['high_inflation']
            inflation_factor = inflation_params.get('inflation_factor', 1.1)
            inflation_volatility = inflation_params.get('volatility_factor', 1.3)
            # インフレ影響（名目値上昇、実質収益率低下）
            inflation_data = base_data * inflation_factor
            inflation_noise = np.random.normal(0, base_std * inflation_volatility, len(base_data))
            scenarios['high_inflation'] = inflation_data + inflation_noise
            
        # 地政学リスクシナリオ
        if 'geopolitical_risk' in stress_scenarios:
            geopolitical_params = stress_scenarios['geopolitical_risk']
            risk_factor = geopolitical_params.get('risk_factor', 0.85)
            risk_volatility = geopolitical_params.get('volatility_factor', 1.8)
            risk_data = base_data * risk_factor
            risk_noise = np.random.normal(0, base_std * risk_volatility, len(base_data))
            scenarios['geopolitical_risk'] = risk_data + risk_noise
            
        return scenarios
        
    def _monte_carlo_simulation(self, base_data: np.ndarray, 
                              forecast_model: Any, steps: int,
                              model_uncertainty: float = 0.1) -> Dict[str, Any]:
        """モンテカルロシミュレーション"""
        logger.info(f"モンテカルロシミュレーション開始: {self.n_simulations}回")
        
        simulations = []
        
        # ベース予測
        try:
            if hasattr(forecast_model, 'forecast'):
                base_forecast = forecast_model.forecast(steps)
                if isinstance(base_forecast, dict):
                    base_prediction = base_forecast.get('forecast', base_forecast.get('ensemble_forecast', []))
                else:
                    base_prediction = base_forecast
            elif hasattr(forecast_model, 'predict'):
                base_predict = forecast_model.predict(base_data.tolist(), steps)
                if isinstance(base_predict, dict):
                    base_prediction = base_predict.get('predictions', [])
                else:
                    base_prediction = base_predict
            else:
                raise ValueError("予測メソッドが見つかりません")
        except Exception as e:
            logger.warning(f"ベース予測失敗: {e}")
            base_prediction = [base_data[-1]] * steps  # フォールバック
            
        base_prediction = np.array(base_prediction[:steps])
        
        # データ統計
        data_std = np.std(base_data)
        data_mean = np.mean(base_data)
        
        # シミュレーション実行
        for sim in range(self.n_simulations):
            try:
                # ランダム要素追加
                
                # 1. モデル不確実性（予測モデルのばらつき）
                model_noise = np.random.normal(0, data_std * model_uncertainty, steps)
                
                # 2. データ不確実性（観測誤差）
                data_noise = np.random.normal(0, data_std * 0.05, steps)
                
                # 3. 構造変化（ランダムショック）
                structural_shock = 0.0
                if np.random.random() < 0.05:  # 5%確率で構造変化
                    shock_magnitude = np.random.normal(0, data_std * 0.3)
                    structural_shock = shock_magnitude
                    
                # 4. 時間的相関（前期の影響）
                temporal_correlation = 0.0
                if sim > 0:
                    prev_sim = simulations[-1]
                    correlation_factor = 0.1
                    temporal_correlation = correlation_factor * (prev_sim[-1] - data_mean)
                    
                # シミュレーション結果
                sim_result = (base_prediction + model_noise + data_noise + 
                            structural_shock + temporal_correlation)
                
                simulations.append(sim_result)
                
            except Exception as e:
                logger.warning(f"シミュレーション{sim}でエラー: {e}")
                # フォールバック: ノイズ付きベース予測
                noise = np.random.normal(0, data_std * 0.1, steps)
                simulations.append(base_prediction + noise)
                
        simulations = np.array(simulations)
        
        # 統計分析
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        simulation_stats = {}
        
        for step in range(steps):
            step_values = simulations[:, step]
            step_stats = {
                'mean': np.mean(step_values),
                'std': np.std(step_values),
                'skewness': stats.skew(step_values),
                'kurtosis': stats.kurtosis(step_values)
            }
            
            # パーセンタイル
            for p in percentiles:
                step_stats[f'percentile_{p}'] = np.percentile(step_values, p)
                
            simulation_stats[f'step_{step+1}'] = step_stats
            
        return {
            'simulations': simulations,
            'n_simulations': self.n_simulations,
            'steps': steps,
            'statistics': simulation_stats,
            'base_prediction': base_prediction.tolist()
        }
        
    def _calculate_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Value at Risk (VaR) 計算"""
        var_results = {}
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_value = np.percentile(returns, alpha * 100)
            var_results[f'var_{int(confidence_level*100)}'] = var_value
            
        # Conditional VaR (Expected Shortfall)
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_threshold = np.percentile(returns, alpha * 100)
            cvar = np.mean(returns[returns <= var_threshold])
            var_results[f'cvar_{int(confidence_level*100)}'] = cvar
            
        return var_results
        
    def _stress_test(self, base_data: np.ndarray, forecast_model: Any,
                    stress_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """ストレステスト"""
        stress_results = {}
        
        # シナリオ生成
        scenarios = self._generate_scenarios(base_data, {'stress_scenarios': stress_scenarios})
        
        for scenario_name, scenario_data in scenarios.items():
            if scenario_name == 'baseline':
                continue
                
            try:
                # ストレスデータでの予測
                if hasattr(forecast_model, 'fit'):
                    # モデル再学習
                    temp_model = type(forecast_model)()
                    for attr in ['max_p', 'max_d', 'max_q', 'seasonal_period', 'n_estimators', 'max_depth']:
                        if hasattr(forecast_model, attr):
                            setattr(temp_model, attr, getattr(forecast_model, attr))
                    temp_model.fit(scenario_data.tolist())
                    
                    if hasattr(temp_model, 'forecast'):
                        stress_forecast = temp_model.forecast(10)
                        if isinstance(stress_forecast, dict):
                            stress_prediction = stress_forecast.get('forecast', stress_forecast.get('ensemble_forecast', []))
                        else:
                            stress_prediction = stress_forecast
                    elif hasattr(temp_model, 'predict'):
                        stress_predict = temp_model.predict(scenario_data.tolist(), 10)
                        if isinstance(stress_predict, dict):
                            stress_prediction = stress_predict.get('predictions', [])
                        else:
                            stress_prediction = stress_predict
                    else:
                        stress_prediction = [scenario_data[-1]] * 10
                else:
                    # 簡易ストレス分析
                    scenario_mean = np.mean(scenario_data)
                    base_mean = np.mean(base_data)
                    stress_factor = scenario_mean / base_mean if base_mean != 0 else 1.0
                    
                    # ベース予測にストレス係数適用
                    if hasattr(forecast_model, 'forecast'):
                        base_forecast = forecast_model.forecast(10)
                        if isinstance(base_forecast, dict):
                            base_prediction = base_forecast.get('forecast', base_forecast.get('ensemble_forecast', []))
                        else:
                            base_prediction = base_forecast
                    else:
                        base_prediction = [base_data[-1]] * 10
                        
                    stress_prediction = np.array(base_prediction) * stress_factor
                    
                # ストレス分析
                scenario_returns = np.diff(scenario_data) / scenario_data[:-1] if len(scenario_data) > 1 else np.array([0])
                stress_var = self._calculate_var(scenario_returns)
                
                stress_results[scenario_name] = {
                    'prediction': stress_prediction.tolist() if hasattr(stress_prediction, 'tolist') else stress_prediction,
                    'scenario_volatility': np.std(scenario_data),
                    'scenario_mean_return': np.mean(scenario_returns) if len(scenario_returns) > 0 else 0,
                    'var_analysis': stress_var,
                    'max_drawdown': self._calculate_max_drawdown(scenario_data)
                }
                
            except Exception as e:
                logger.warning(f"ストレステスト{scenario_name}でエラー: {e}")
                stress_results[scenario_name] = {'error': str(e)}
                
        return stress_results
        
    def _calculate_max_drawdown(self, data: np.ndarray) -> float:
        """最大ドローダウン計算"""
        if len(data) < 2:
            return 0.0
            
        peak = data[0]
        max_dd = 0.0
        
        for value in data[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
        return max_dd
        
    def _sensitivity_analysis(self, base_data: np.ndarray, forecast_model: Any,
                            parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """感度分析"""
        sensitivity_results = {}
        
        # ベース予測
        try:
            if hasattr(forecast_model, 'forecast'):
                base_forecast = forecast_model.forecast(5)
                if isinstance(base_forecast, dict):
                    base_value = np.mean(base_forecast.get('forecast', base_forecast.get('ensemble_forecast', [0])))
                else:
                    base_value = np.mean(base_forecast)
            else:
                base_value = base_data[-1]
        except:
            base_value = base_data[-1]
            
        # パラメータ感度分析
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_impacts = []
            param_values = np.linspace(min_val, max_val, 10)
            
            for param_val in param_values:
                try:
                    # パラメータ変更シナリオ
                    if param_name == 'volatility':
                        # ボラティリティ変更
                        vol_factor = param_val
                        scenario_data = base_data + np.random.normal(0, np.std(base_data) * vol_factor, len(base_data))
                    elif param_name == 'trend':
                        # トレンド変更
                        trend_factor = param_val
                        scenario_data = base_data * trend_factor
                    elif param_name == 'correlation':
                        # 自己相関変更（簡易実装）
                        corr_factor = param_val
                        scenario_data = base_data.copy()
                        for i in range(1, len(scenario_data)):
                            scenario_data[i] += corr_factor * (scenario_data[i-1] - np.mean(base_data))
                    else:
                        scenario_data = base_data
                        
                    # シナリオ予測
                    if len(scenario_data) == len(base_data):
                        # 簡易予測（最終値ベース）
                        scenario_value = scenario_data[-1]
                    else:
                        scenario_value = base_value
                        
                    impact = (scenario_value - base_value) / base_value if base_value != 0 else 0
                    param_impacts.append(impact)
                    
                except Exception as e:
                    logger.warning(f"感度分析{param_name}={param_val}でエラー: {e}")
                    param_impacts.append(0.0)
                    
            sensitivity_results[param_name] = {
                'parameter_values': param_values.tolist(),
                'impact_values': param_impacts,
                'sensitivity': np.std(param_impacts),  # 感度指標
                'max_impact': max(param_impacts),
                'min_impact': min(param_impacts)
            }
            
        return sensitivity_results
        
    def analyze_scenarios(self, data: List[float], forecast_model: Any,
                         scenario_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        包括的シナリオ分析
        
        Args:
            data: 時系列データ
            forecast_model: 予測モデル
            scenario_config: シナリオ設定
            
        Returns:
            シナリオ分析結果
        """
        if scenario_config is None:
            scenario_config = {
                'monte_carlo': True,
                'stress_test': True,
                'sensitivity_analysis': True,
                'forecast_steps': 10,
                'model_uncertainty': 0.1
            }
            
        logger.info("シナリオ分析開始")
        
        data_array = np.array(data)
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_size': len(data),
            'scenario_config': scenario_config
        }
        
        # モンテカルロシミュレーション
        if scenario_config.get('monte_carlo', True):
            try:
                logger.info("モンテカルロシミュレーション実行")
                mc_results = self._monte_carlo_simulation(
                    data_array, forecast_model,
                    scenario_config.get('forecast_steps', 10),
                    scenario_config.get('model_uncertainty', 0.1)
                )
                results['monte_carlo'] = mc_results
                
                # VaR分析
                final_step_values = mc_results['simulations'][:, -1]
                final_returns = (final_step_values - data_array[-1]) / data_array[-1]
                var_analysis = self._calculate_var(final_returns)
                results['var_analysis'] = var_analysis
                
            except Exception as e:
                logger.warning(f"モンテカルロシミュレーション失敗: {e}")
                results['monte_carlo'] = {'error': str(e)}
                
        # ストレステスト
        if scenario_config.get('stress_test', True):
            try:
                logger.info("ストレステスト実行")
                stress_scenarios = scenario_config.get('stress_scenarios', {
                    'financial_crisis': {'drop_factor': 0.6, 'volatility_factor': 2.0},
                    'high_inflation': {'inflation_factor': 1.1, 'volatility_factor': 1.3},
                    'geopolitical_risk': {'risk_factor': 0.85, 'volatility_factor': 1.8}
                })
                stress_results = self._stress_test(data_array, forecast_model, stress_scenarios)
                results['stress_test'] = stress_results
                
            except Exception as e:
                logger.warning(f"ストレステスト失敗: {e}")
                results['stress_test'] = {'error': str(e)}
                
        # 感度分析
        if scenario_config.get('sensitivity_analysis', True):
            try:
                logger.info("感度分析実行")
                parameter_ranges = scenario_config.get('parameter_ranges', {
                    'volatility': (0.5, 2.0),
                    'trend': (0.8, 1.2),
                    'correlation': (-0.5, 0.5)
                })
                sensitivity_results = self._sensitivity_analysis(
                    data_array, forecast_model, parameter_ranges
                )
                results['sensitivity_analysis'] = sensitivity_results
                
            except Exception as e:
                logger.warning(f"感度分析失敗: {e}")
                results['sensitivity_analysis'] = {'error': str(e)}
                
        # リスクサマリー
        results['risk_summary'] = self._generate_risk_summary(results)
        
        self.simulation_results = results
        return results
        
    def _generate_risk_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """リスクサマリー生成"""
        summary = {
            'overall_risk_level': 'medium',  # デフォルト
            'key_risks': [],
            'risk_metrics': {}
        }
        
        # VaR分析からのリスク評価
        if 'var_analysis' in analysis_results:
            var_data = analysis_results['var_analysis']
            var_95 = var_data.get('var_95', 0)
            
            if var_95 < -0.2:  # 20%以上の損失リスク
                summary['key_risks'].append('高い下落リスク（VaR95% > 20%）')
                summary['overall_risk_level'] = 'high'
            elif var_95 < -0.1:
                summary['key_risks'].append('中程度の下落リスク（VaR95% > 10%）')
                
            summary['risk_metrics']['var_95'] = var_95
            
        # ストレステストからのリスク評価
        if 'stress_test' in analysis_results:
            stress_data = analysis_results['stress_test']
            stress_risks = []
            
            for scenario, result in stress_data.items():
                if isinstance(result, dict) and 'max_drawdown' in result:
                    max_dd = result['max_drawdown']
                    if max_dd > 0.3:  # 30%以上のドローダウン
                        stress_risks.append(f'{scenario}: 最大ドローダウン{max_dd:.1%}')
                        
            if stress_risks:
                summary['key_risks'].extend(stress_risks)
                if len(stress_risks) >= 2:
                    summary['overall_risk_level'] = 'high'
                    
        # 感度分析からのリスク評価
        if 'sensitivity_analysis' in analysis_results:
            sens_data = analysis_results['sensitivity_analysis']
            high_sensitivity_params = []
            
            for param, result in sens_data.items():
                if isinstance(result, dict) and 'sensitivity' in result:
                    sensitivity = result['sensitivity']
                    if sensitivity > 0.1:  # 高感度
                        high_sensitivity_params.append(f'{param}({sensitivity:.2f})')
                        
            if high_sensitivity_params:
                summary['key_risks'].append(f'高感度パラメータ: {", ".join(high_sensitivity_params)}')
                
        # 総合リスクレベル調整
        if len(summary['key_risks']) == 0:
            summary['overall_risk_level'] = 'low'
        elif len(summary['key_risks']) >= 4:
            summary['overall_risk_level'] = 'high'
            
        return summary
        
    def get_scenario_report(self) -> Dict[str, Any]:
        """シナリオ分析レポート生成"""
        if not self.simulation_results:
            return {'error': '分析結果がありません'}
            
        report = {
            'executive_summary': self._create_executive_summary(),
            'detailed_analysis': self.simulation_results,
            'recommendations': self._generate_recommendations(),
            'report_timestamp': datetime.now().isoformat()
        }
        
        return report
        
    def _create_executive_summary(self) -> Dict[str, Any]:
        """エグゼクティブサマリー"""
        if not self.simulation_results:
            return {}
            
        summary = {}
        
        # リスクレベル
        risk_summary = self.simulation_results.get('risk_summary', {})
        summary['overall_risk_assessment'] = risk_summary.get('overall_risk_level', 'unknown')
        summary['key_risk_factors'] = risk_summary.get('key_risks', [])
        
        # 主要指標
        if 'var_analysis' in self.simulation_results:
            var_data = self.simulation_results['var_analysis']
            summary['value_at_risk_95'] = var_data.get('var_95', 'N/A')
            summary['expected_shortfall_95'] = var_data.get('cvar_95', 'N/A')
            
        # モンテカルロ結果
        if 'monte_carlo' in self.simulation_results:
            mc_data = self.simulation_results['monte_carlo']
            if 'statistics' in mc_data:
                final_step = f"step_{mc_data.get('steps', 1)}"
                if final_step in mc_data['statistics']:
                    final_stats = mc_data['statistics'][final_step]
                    summary['forecast_confidence_interval'] = {
                        'mean': final_stats.get('mean'),
                        '5th_percentile': final_stats.get('percentile_5'),
                        '95th_percentile': final_stats.get('percentile_95')
                    }
                    
        return summary
        
    def _generate_recommendations(self) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if not self.simulation_results:
            return recommendations
            
        risk_level = self.simulation_results.get('risk_summary', {}).get('overall_risk_level', 'medium')
        
        if risk_level == 'high':
            recommendations.extend([
                'リスク管理戦略の見直しを推奨',
                'ポートフォリオの多様化を検討',
                'ストップロス戦略の実装を検討'
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                '定期的なリスク監視を継続',
                '市場変動への対応プランを準備'
            ])
        else:
            recommendations.append('現在のリスクレベルは許容範囲内')
            
        # VaR分析に基づく推奨
        if 'var_analysis' in self.simulation_results:
            var_95 = self.simulation_results['var_analysis'].get('var_95', 0)
            if var_95 < -0.15:
                recommendations.append('ヘッジ戦略の導入を検討')
                
        return recommendations