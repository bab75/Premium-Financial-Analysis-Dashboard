"""
HTML Report Generator for Financial Analysis
Creates comprehensive downloadable HTML reports with interactive charts and predictive price tables
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import json
import os

class HTMLReportGenerator:
    """Generate comprehensive HTML reports with interactive charts, predictive tables, and modern UI."""
    
    def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to resolve index/column ambiguity issues."""
        if data is None:
            return data
        
        data = data.copy()
        
        try:
            if 'Datetime' in data.columns:
                if isinstance(data.index, pd.DatetimeIndex) or (
                    hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype).lower()
                ):
                    data = data.drop(columns=['Datetime'], errors='ignore')
                else:
                    try:
                        data = data.set_index('Datetime')
                    except:
                        data = data.drop(columns=['Datetime'], errors='ignore')
            
            if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                try:
                    if data['Date'].dtype != 'int64':
                        data = data.set_index('Date')
                        data.index = pd.to_datetime(data.index)
                except:
                    pass
            
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index(drop=True)
                
        except Exception:
            try:
                data = data.reset_index(drop=True)
                for col in ['Datetime', 'Date']:
                    if col in data.columns and data[col].dtype == 'int64':
                        data = data.drop(columns=[col], errors='ignore')
            except:
                pass
        
        return data
    
    def __init__(self):
        self.css_styles = """
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                body { font-family: 'Inter', sans-serif; }
                .card { background: white; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .table-responsive { overflow-x: auto; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 0.75rem; text-align: left; }
                th { background: #f3f4f6; }
                tr:nth-child(even) { background: #f9fafb; }
                tr:hover { background: #f1f5f9; }
                .signal-buy { color: #15803d; }
                .signal-sell { color: #b91c1c; }
                .signal-hold { color: #d97706; }
            </style>
        """
    
    def generate_comprehensive_report(self, 
                                    stock_symbol: str,
                                    historical_data: pd.DataFrame,
                                    tech_indicators,
                                    analytics,
                                    visualizations,
                                    prediction_data: Optional[Dict] = None,
                                    prediction_days: Optional[int] = None,
                                    advanced_analytics=None,
                                    report_type: str = "full") -> str:
        """Generate a comprehensive HTML report with all analysis components."""
        
        historical_data = self._clean_dataframe(historical_data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = "<html><head><title>Financial Analysis Report - " + stock_symbol + "</title>" + self.css_styles + "</head><body class='bg-gray-100'>"
        html_content += "<div class='container mx-auto p-4 max-w-7xl'>"
        
        html_content += "<header class='text-center mb-8'><h1 class='text-3xl font-bold text-gray-800'>Financial Analysis Report</h1>"
        html_content += "<p class='text-gray-600'><strong>Stock Symbol:</strong> " + stock_symbol + " | <strong>Generated:</strong> " + timestamp + "</p></header>"
        
        html_content += "<main class='space-y-8'>"
        html_content += self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)
        html_content += self._generate_technical_charts_section(tech_indicators)
        html_content += self._generate_trading_signals_section(tech_indicators)
        html_content += self._generate_price_charts_section(visualizations)
        html_content += self._generate_performance_metrics(historical_data)
        html_content += self._generate_risk_analysis(analytics, historical_data)
        
        if prediction_data is not None and prediction_days is not None and report_type in ["full", "predictions"]:
            html_content += self._generate_prediction_charts_section(stock_symbol, historical_data, prediction_data, prediction_days)
        
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_3d_charts_section(visualizations)
        
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_advanced_analytics_section(advanced_analytics)
        
        html_content += "</main>"
        html_content += "<footer class='mt-8 text-center text-gray-600'><p>This report was generated automatically by the Financial Analysis Application</p>"
        html_content += "<p><strong>Data analysis period:</strong> " + str(historical_data.index[0])[:10] + " to " + str(historical_data.index[-1])[:10] + "</p></footer>"
        html_content += "</div></body></html>"
        
        return html_content
    
    def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
        """Generate executive summary section."""
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
        high_52week = data['High'].max()
        low_52week = data['Low'].min()
        avg_volume = data['Volume'].mean()
        
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>📊 Executive Summary</h2><div class='table-responsive'><table class='table-auto'>"
        html_section += "<tr><th>Metric</th><th>Value</th></tr>"
        html_section += "<tr><td>Current Price</td><td>$" + str(round(current_price, 2)) + "</td></tr>"
        html_section += "<tr><td>Daily Change</td><td>" + f"{price_change:+.2f} ({price_change_pct:+.2f}%)" + "</td></tr>"
        html_section += "<tr><td>52-Week High</td><td>$" + str(round(high_52week, 2)) + "</td></tr>"
        html_section += "<tr><td>52-Week Low</td><td>$" + str(round(low_52week, 2)) + "</td></tr>"
        html_section += "<tr><td>Average Volume</td><td>" + f"{avg_volume:,.0f}" + "</td></tr>"
        html_section += "</table></div></section>"
        
        return html_section
    
    def _generate_technical_charts_section(self, tech_indicators) -> str:
        """Generate technical analysis charts section."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>📈 Technical Analysis Charts</h2><p class='text-gray-600 mb-4'>Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.</p>"
        
        try:
            ma_chart = tech_indicators.create_moving_averages_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Moving Averages</h3>" + ma_chart.to_html(include_plotlyjs='inline', div_id='ma-chart')
            
            rsi_chart = tech_indicators.create_rsi_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Relative Strength Index (RSI)</h3>" + rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')
            
            macd_chart = tech_indicators.create_macd_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>MACD Analysis</h3>" + macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')
            
            bb_chart = tech_indicators.create_bollinger_bands_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Bollinger Bands</h3>" + bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating technical charts: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def _generate_trading_signals_section(self, tech_indicators) -> str:
        """Generate trading signals analysis section."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>🎯 Trading Signals & Recommendations</h2>"
        
        try:
            signals = tech_indicators.get_trading_signals()
            
            for indicator, signal_data in signals.items():
                signal_type = signal_data.get('signal', 'Unknown').lower()
                strength = signal_data.get('strength', 'Unknown')
                
                signal_class = 'signal-hold'
                if 'buy' in signal_type:
                    signal_class = 'signal-buy'
                elif 'sell' in signal_type:
                    signal_class = 'signal-sell'
                
                html_section += "<div class='mb-4'><h3 class='text-xl font-medium text-gray-700'>" + indicator + "</h3>"
                html_section += "<p>Signal: <span class='" + signal_class + "'>" + signal_data.get('signal', 'Unknown') + "</span></p>"
                html_section += "<p>Strength: " + strength + "</p></div>"
                
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating trading signals: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def _generate_price_charts_section(self, visualizations) -> str:
        """Generate price visualization charts section."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>💹 Price Analysis Charts</h2>"
        
        try:
            candlestick_chart = visualizations.create_candlestick_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Candlestick Chart</h3>" + candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')
            
            trends_chart = visualizations.create_price_trends_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Price Trends</h3>" + trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')
            
            volume_chart = visualizations.create_volume_chart()
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Volume Analysis</h3>" + volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating price charts: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
        """Generate performance metrics table."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>📊 Performance Metrics</h2><div class='table-responsive'><table class='table-auto'>"
        html_section += "<tr><th>Metric</th><th>Value</th><th>Description</th></tr>"
        
        try:
            total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
            max_price = data['High'].max()
            min_price = data['Low'].min()
            avg_volume = data['Volume'].mean()
            
            metrics = [
                ("Total Return", f"{total_return:.2f}%", "Overall price appreciation/depreciation"),
                ("Annualized Volatility", f"{volatility:.2f}%", "Price volatility over the period"),
                ("Maximum Price", f"${max_price:.2f}", "Highest price reached"),
                ("Minimum Price", f"${min_price:.2f}", "Lowest price reached"),
                ("Average Volume", f"{avg_volume:,.0f}", "Average daily trading volume"),
                ("Data Points", f"{len(data)}", "Number of trading days analyzed")
            ]
            
            for metric, value, description in metrics:
                html_section += "<tr><td>" + metric + "</td><td>" + value + "</td><td>" + description + "</td></tr>"
                
        except Exception as e:
            html_section += "<tr><td colspan='3' class='text-red-600'>Error calculating metrics: " + str(e) + "</td></tr>"
        
        html_section += "</table></div></section>"
        return html_section
    
    def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>⚠️ Risk Analysis</h2>"
        
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
            
            html_section += "<div class='table-responsive'><table class='table-auto'><tr><th>Metric</th><th>Value</th></tr>"
            html_section += "<tr><td>Annualized Volatility</td><td>" + f"{volatility:.2f}%" + "</td></tr>"
            html_section += "<tr><td>Maximum Drawdown</td><td>" + f"{max_drawdown:.2f}%" + "</td></tr></table></div>"
            html_section += "<h3 class='text-xl font-medium text-gray-700 mt-4'>Risk Assessment</h3>"
            
            if volatility > 30:
                html_section += "<p class='text-red-600'>🔴 High Risk: This stock shows high volatility. Suitable for experienced traders with high risk tolerance.</p>"
            elif volatility > 20:
                html_section += "<p class='text-yellow-600'>🟡 Medium Risk: Moderate volatility. Suitable for balanced investment strategies.</p>"
            else:
                html_section += "<p class='text-green-600'>🟢 Low Risk: Relatively stable price movements. Suitable for conservative investors.</p>"
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error calculating risk metrics: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "financial_analysis_report_" + timestamp + ".html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _save_predictions(self, stock_symbol: str, predictions: Dict) -> None:
        """Save predictions to a JSON file."""
        try:
            filename = f"{stock_symbol}_predictions.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, default=str)
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def _load_predictions(self, stock_symbol: str) -> Optional[Dict]:
        """Load predictions from a JSON file."""
        try:
            filename = f"{stock_symbol}_predictions.json"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading predictions: {e}")
        return None
    
    def _generate_prediction_charts_section(self, stock_symbol: str, historical_data: pd.DataFrame, prediction_data: Dict, prediction_days: int) -> str:
        """Generate prediction charts and price table section for HTML report."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>📈 Price Predictions</h2><p class='text-gray-600 mb-4'>Price predictions based on " + prediction_data.get('method', 'unknown method').replace('_', ' ').title() + " for the next " + str(prediction_days) + " days.</p>"
        
        try:
            import pandas as pd
            
            # Load or use provided prediction data
            saved_predictions = self._load_predictions(stock_symbol)
            if saved_predictions is None or prediction_data.get('method') not in saved_predictions:
                # Save new predictions
                predictions_to_save = {
                    prediction_data['method']: {
                        'prices': [float(p) for p in prediction_data['prices']],
                        'dates': [(pd.Timestamp(historical_data.index[-1]) + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(prediction_days)],
                        'description': 'Prediction based on ' + prediction_data['method'],
                        'confidence': prediction_data.get('confidence', {}),
                        'disclaimer': prediction_data.get('disclaimer', '')
                    }
                }
                self._save_predictions(stock_symbol, predictions_to_save)
                pred_data = prediction_data['prices']
                pred_dates = predictions_to_save[prediction_data['method']]['dates']
            else:
                pred_data = saved_predictions[prediction_data['method']]['prices']
                pred_dates = saved_predictions[prediction_data['method']]['dates']
            
            # Generate prediction chart
            recent_data = historical_data.tail(20)
            historical_dates = recent_data.index
            historical_prices = recent_data['Close'].values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_prices,
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_data,
                mode='lines+markers',
                name='Predicted Prices',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title='Price Prediction - ' + prediction_data.get('method', 'Unknown').replace('_', ' ').title(),
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            chart_html = fig.to_html(include_plotlyjs=False, div_id="prediction_" + prediction_data.get('method', 'unknown'))
            html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>" + prediction_data.get('method', 'Unknown').replace('_', ' ').title() + "</h3>"
            html_section += "<p class='text-gray-600 mb-4'>Prediction for the next " + str(prediction_days) + " days.</p>"
            html_section += chart_html
            
            # Generate predictive price table
            html_section += "<h3 class='text-xl font-medium text-gray-700 mt-6 mb-2'>Predicted Prices</h3>"
            html_section += "<div class='table-responsive'><table class='table-auto'>"
            html_section += "<tr><th>Date</th><th>Day</th><th>Predicted Price</th></tr>"
            
            for i in range(prediction_days):
                html_section += "<tr><td>" + pred_dates[i] + "</td><td>Day " + str(i+1) + "</td><td>$" + f"{pred_data[i]:.2f}" + "</td></tr>"
            
            html_section += "</table></div>"
            
            # Add prediction metrics
            confidence = prediction_data.get('confidence', {})
            disclaimer = prediction_data.get('disclaimer', '')
            
            metrics = [
                ("Confidence Score", confidence.get('score', 'N/A'), "/10"),
                ("Volatility", confidence.get('volatility', 'N/A'), "%"),
                ("Trend Strength", confidence.get('trend_strength', 'N/A'), ""),
                ("Data Quality", confidence.get('data_quality', 'N/A'), "")
            ]
            
            html_section += "<h3 class='text-xl font-medium text-gray-700 mt-6 mb-2'>Prediction Metrics</h3>"
            html_section += "<div class='table-responsive'><table class='table-auto'><tr><th>Metric</th><th>Value</th></tr>"
            
            for metric, value, suffix in metrics:
                formatted_value = f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else str(value)
                html_section += "<tr><td>" + metric + "</td><td>" + formatted_value + "</td></tr>"
            
            html_section += "</table></div>"
            html_section += "<p class='text-gray-600 italic mt-4'>" + disclaimer + "</p>"
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating predictions: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def _generate_3d_charts_section(self, visualizations) -> str:
        """Generate 3D visualization charts section for HTML report."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>📊 3D Visualizations</h2><p class='text-gray-600 mb-4'>Interactive three-dimensional analysis for comprehensive market insights.</p>"
        
        try:
            if hasattr(visualizations, 'get_3d_price_volume_chart'):
                chart = visualizations.get_3d_price_volume_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_price_volume")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>3D Price-Volume Analysis</h3><p class='text-gray-600 mb-4'>Three-dimensional visualization of price movements, volume, and time relationships.</p>" + chart_html
            
            if hasattr(visualizations, 'get_3d_technical_surface'):
                chart = visualizations.get_3d_technical_surface()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_technical_surface")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>3D Technical Indicator Surface</h3><p class='text-gray-600 mb-4'>Surface plot showing relationships between multiple technical indicators.</p>" + chart_html
            
            if hasattr(visualizations, 'get_3d_market_dynamics'):
                chart = visualizations.get_3d_market_dynamics()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_market_dynamics")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>3D Market Dynamics</h3><p class='text-gray-600 mb-4'>Multi-dimensional view of market behavior and trading patterns.</p>" + chart_html
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating 3D charts: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
    
    def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
        """Generate advanced analytics section for HTML report."""
        html_section = "<section class='card p-6'><h2 class='text-2xl font-semibold text-gray-800 mb-4'>🎯 Advanced Analytics</h2><p class='text-gray-600 mb-4'>Comprehensive sector analysis, correlations, and market intelligence.</p>"
        
        try:
            if hasattr(advanced_analytics, 'create_sector_performance_chart'):
                chart = advanced_analytics.create_sector_performance_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="sector_performance")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Sector Performance Analysis</h3><p class='text-gray-600 mb-4'>Comparative performance across different market sectors.</p>" + chart_html
            
            if hasattr(advanced_analytics, 'create_correlation_heatmap'):
                chart = advanced_analytics.create_correlation_heatmap()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="correlation_heatmap")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Market Correlation Analysis</h3><p class='text-gray-600 mb-4'>Heat map showing correlations between different market metrics and indicators.</p>" + chart_html
            
            if hasattr(advanced_analytics, 'create_performance_dashboard'):
                chart = advanced_analytics.create_performance_dashboard()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="performance_dashboard")
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Comprehensive Performance Dashboard</h3><p class='text-gray-600 mb-4'>Multi-metric dashboard showing key performance indicators and trends.</p>" + chart_html
            
            if hasattr(advanced_analytics, 'get_industry_analysis'):
                industry_data = advanced_analytics.get_industry_analysis()
                if industry_data is not None and not industry_data.empty:
                    html_section += "<h3 class='text-xl font-medium text-gray-700 mb-2'>Industry Analysis Summary</h3><div class='table-responsive'><table class='table-auto'>"
                    html_section += industry_data.head(10).to_html(classes="summary-table", escape=False)
                    html_section += "</table></div>"
            
        except Exception as e:
            html_section += "<p class='text-red-600'>Error generating advanced analytics: " + str(e) + "</p>"
        
        html_section += "</section>"
        return html_section
