"""
HTML Report Generator for Financial Analysis
Creates comprehensive downloadable HTML reports with interactive charts
"""

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
import pytz
from typing import Dict, List, Optional
import base64
import html
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class HTMLReportGenerator:
    """Generate comprehensive HTML reports with interactive charts and analysis."""
    
    def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to resolve index/column ambiguity issues."""
        if data is None or data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        try:
            print("Input DataFrame columns:", data.columns.tolist())
            print("Input DataFrame index type:", type(data.index).__name__)
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
                if data['Datetime'].isna().any():
                    print("Warning: Dropping rows with invalid Datetime values")
                    data = data.dropna(subset=['Datetime'])
                data = data.set_index('Datetime')
            elif data.index.name == 'Datetime' or 'Date' in data.columns:
                if 'Date' in data.columns:
                    data['Datetime'] = pd.to_datetime(data['Date'], errors='coerce')
                    data = data.drop(columns=['Date'])
                data['Datetime'] = pd.to_datetime(data.index, errors='coerce')
                if data['Datetime'].isna().all():
                    print("Creating fallback Datetime index due to missing valid dates")
                    data['Datetime'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                data = data.set_index('Datetime')
            else:
                print("No Datetime column or index found, creating fallback Datetime index")
                data['Datetime'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                data = data.set_index('Datetime')
            
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close'] if 'Close' in data.columns else 0
            for col in ['Dividends', 'Stock Splits']:
                if col not in data.columns:
                    data[col] = 0
            data = data.sort_index()
            print("Cleaned DataFrame index type:", type(data.index).__name__)
            print("Cleaned DataFrame first and last dates:", data.index[0], data.index[-1])
            return data
        except Exception as e:
            logging.error(f"Data cleaning failed: {str(e)}")
            return data.reset_index(drop=True)

    def __init__(self):
        self.css_styles = """
            <style>
                body {
                    font-family: 'Arial', 'Helvetica', sans-serif;
                    margin: 20px;
                    background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
                    font-size: 16px;
                    color: #2d3748;
                    line-height: 1.6;
                }
                h1 {
                    color: #2b6cb0;
                    text-align: center;
                    font-size: 2.5em;
                    font-weight: 700;
                    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
                    background: linear-gradient(90deg, #2b6cb0, #63b3ed);
                    -webkit-background-clip: text;
                    background-clip: text;
                    color: transparent;
                    margin-bottom: 20px;
                    animation: fadeIn 1s ease-in;
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                h2, h3, h4 {
                    color: #2c5282;
                    font-weight: 600;
                    margin-top: 20px;
                }
                h2 { font-size: 1.8em; }
                h3 { font-size: 1.5em; }
                h4 { font-size: 1.2em; }
                .section {
                    margin: 25px 0;
                    background: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                .section:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
                }
                .metric-card {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }
                .metric {
                    background: linear-gradient(145deg, #ffffff, #edf2f7);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    transition: all 0.3s ease;
                    cursor: default;
                }
                .metric:hover {
                    transform: scale(1.03);
                    background: linear-gradient(145deg, #edf2f7, #e2e8f0);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                    background: #fff;
                    font-size: 0.95em;
                }
                th, td {
                    border: 1px solid #e2e8f0;
                    padding: 10px;
                    text-align: left;
                }
                th {
                    background: linear-gradient(90deg, #2b6cb0, #63b3ed);
                    color: #ffffff;
                    font-weight: 600;
                }
                .disclaimer {
                    font-style: italic;
                    color: #4a5568;
                    font-size: 0.9em;
                    margin-top: 20px;
                }
                .signal-buy { color: #48bb78; font-weight: bold; }
                .signal-sell { color: #f56565; font-weight: bold; }
                .signal-hold { color: #718096; font-weight: bold; }
                details {
                    margin: 10px 0;
                    background: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    overflow: hidden;
                }
                summary {
                    cursor: pointer;
                    font-weight: 600;
                    color: #2c5282;
                    padding: 12px;
                    background: linear-gradient(90deg, #f0f4f8, #d9e2ec);
                    transition: all 0.3s ease;
                }
                summary:hover {
                    background: linear-gradient(90deg, #d9e2ec, #f0f4f8);
                    color: #2b6cb0;
                }
                details[open] summary {
                    background: linear-gradient(90deg, #2b6cb0, #63b3ed);
                    color: #ffffff;
                }
                details div {
                    padding: 15px;
                    background: #ffffff;
                }
                .chart-container {
                    margin: 15px 0;
                    padding: 10px;
                    background: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    transition: transform 0.3s ease;
                }
                .chart-container:hover {
                    transform: translateY(-2px);
                }
            </style>
        """
        self.js_script = """
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        """

    def generate_comprehensive_report(self, 
                                    stock_symbol: str,
                                    historical_data: pd.DataFrame,
                                    tech_indicators,
                                    analytics,
                                    visualizations,
                                    predictions=None,
                                    advanced_analytics=None,
                                    additional_figures=None,
                                    report_type: str = "full") -> str:
        """Generate a comprehensive HTML report with all analysis components."""
        logging.info(f"Generating report for {stock_symbol}, report_type: {report_type}")
        
        historical_data = self._clean_dataframe(historical_data)
        logging.debug(f"Cleaned historical_data index: {historical_data.index}")

        local_tz = pytz.timezone('America/New_York')  # EDT timezone
        timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        stock_symbol = html.escape(str(stock_symbol))
        timestamp = html.escape(str(timestamp))
        
        if historical_data.empty:
            return f"""
            <html>
            <head><title>Financial Analysis Report - {stock_symbol}</title>{self.css_styles}{self.js_script}</head>
            <body>
                <h1>Financial Analysis Report: {stock_symbol}</h1>
                <p style="text-align: center;">Generated on: {timestamp}</p>
                <div class="section"><p>No historical data available to generate the report.</p></div>
            </body>
            </html>
            """.encode('utf-8')

        html_content = f"""
        <html>
        <head>
            <title>Financial Analysis Report - {stock_symbol}</title>
            {self.css_styles}
            {self.js_script}
        </head>
        <body>
            <h1>Financial Analysis Report: {stock_symbol}</h1>
            <p style="text-align: center;">Generated on: {timestamp}</p>
            <div class="section">
                <h2>Overview</h2>
                {self._generate_overview(historical_data)}
            </div>
            <div class="section">
                <h2>Advanced Visualizations</h2>
                {self._generate_visualizations(additional_figures)}
            </div>
            <div class="section">
                <h2>Technical Indicators</h2>
                {self._generate_technical_indicators(tech_indicators)}
            </div>
            <div class="section">
                <h2>Trading Insights</h2>
                {self._generate_trading_insights(tech_indicators, analytics)}
            </div>
        """
        
        if predictions is not None and len(historical_data) > 50 and report_type in ["full", "predictions"]:
            html_content += f"""
            <div class="section">
                <h2>Price Predictions</h2>
                {self._generate_price_predictions(historical_data, predictions)}
            </div>
            """
        
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += f"""
            <div class="section">
                <h2>Comparative Analysis</h2>
                {self._generate_comparative_analysis(advanced_analytics)}
            </div>
            """
        
        if not historical_data.empty:
            html_content += f"""
            <div class="section">
                <h2>Data Summary</h2>
                {self._generate_data_summary(historical_data)}
            </div>
            """
        
        html_content += f"""
            <div class="section disclaimer">
                <p><strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute financial advice. Always conduct your own research before making investment decisions.</p>
                <p>Data analysis period: {historical_data.index[0].strftime('%Y-%m-%d') if not historical_data.empty else 'N/A'} to {historical_data.index[-1].strftime('%Y-%m-%d') if not historical_data.empty else 'N/A'}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content.encode('utf-8')

    def _generate_overview(self, data: pd.DataFrame) -> str:
        """Generate overview section."""
        html_section = ""
        try:
            if not data.empty:
                start_date = data.index[0].strftime('%Y-%m-%d')
                end_date = data.index[-1].strftime('%Y-%m-%d')
                current_price = data['Close'].iloc[-1] if 'Close' in data.columns else 0
                total_return = ((current_price / data['Close'].iloc[0]) - 1) * 100 if len(data) > 1 and 'Close' in data.columns else 0
                volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100 if len(data) > 1 and 'Close' in data.columns else 0
                
                html_section = f"""
                <div class="metric-card">
                    <div class="metric"><strong>Period</strong><br>{start_date} to {end_date}</div>
                    <div class="metric"><strong>Current Price</strong><br>${current_price:.2f}</div>
                    <div class="metric"><strong>Total Return</strong><br>{total_return:.2f}%</div>
                    <div class="metric"><strong>Volatility (Annual)</strong><br>{volatility:.2f}%</div>
                </div>
                """
            else:
                html_section = "<p>No historical data available for overview.</p>"
        except Exception as e:
            html_section = f"<p>Error generating overview: {html.escape(str(e))}</p>"
        return html_section

    def _generate_visualizations(self, additional_figures: Dict) -> str:
        """Generate advanced visualizations section with optimized size and error handling."""
        html_section = "<p>Interactive charts for advanced analysis.</p>"
        try:
            chart_mappings = [
                ('candlestick', 'Candlestick Chart', 'candlestick-chart'),
                ('price_trends', 'Price Trends', 'price-trends-chart'),
                ('volume_analysis', 'Volume Analysis', 'volume-chart'),
                ('3d_factor_analysis', '3D Factor Analysis', '3d-factor-chart'),
                ('3d_risk_surface', '3D Risk Surface Analysis', '3d-risk-surface-chart'),
                ('market_overview_dashboard', 'Market Overview Dashboard', 'market-overview-chart'),
                ('market_cap_chart', 'Market Capitalization Chart', 'market-cap-chart'),
                ('sector_pie_chart', 'Sector Distribution', 'sector-pie-chart'),
                ('correlation_heatmap_daily', 'Daily Correlation Heatmap', 'corr-heatmap-daily'),
                ('performance_scatter', 'Performance vs Volume Scatter', 'perf-scatter-chart'),
                ('advanced_candlestick', 'Advanced Candlestick Chart', 'adv-candlestick-chart'),
                ('correlation_heatmap', 'Correlation Heatmap', 'corr-heatmap-chart'),
                ('risk_gauge', 'Risk Gauge', 'risk-gauge-chart'),
                ('volatility_gauge', 'Volatility Gauge', 'volatility-gauge-chart'),
                ('performance_gauge', 'Performance Gauge', 'performance-gauge-chart'),
                ('advanced_dashboard', 'Advanced Multi-Metric Dashboard', 'advanced-dashboard-chart'),
                ('moving_averages', 'Moving Averages', 'ma-chart'),
                ('rsi_chart', 'RSI Chart', 'rsi-chart'),
                ('macd_chart', 'MACD Chart', 'macd-chart'),
                ('bollinger_bands', 'Bollinger Bands', 'bb-chart'),
            ]
            colors = ['#2b6cb0', '#48bb78', '#f56565', '#ecc94b', '#9f7aea', '#ed8936', '#38b2ac', '#667eea', '#ed64a6', '#d69e2e']
            for key, title, chart_id in chart_mappings:
                if key in additional_figures and additional_figures[key]:
                    fig = additional_figures[key]
                    # Optimize chart size
                    fig.update_layout(height=300, width=500, margin=dict(l=30, r=30, t=30, b=30), hovermode='closest')
                    # Apply colors based on chart type
                    if fig.data and hasattr(fig.data[0], 'type'):
                        if fig.data[0].type == 'pie':
                            fig.update_traces(marker=dict(colors=colors[:len(fig.data[0].labels)] if len(fig.data[0].labels) <= len(colors) else colors))
                        elif fig.data[0].type == 'bar':
                            fig.update_traces(marker_color=colors[0] if len(fig.data) == 1 else [colors[i % len(colors)] for i in range(len(fig.data[0].x))])
                        elif fig.data[0].type in ['scatter', 'candlestick']:
                            fig.update_traces(line_color=colors[0], marker_color=colors[0])
                    chart_html = pio.to_html(fig, full_html=False, config={'displayModeBar': False, 'responsive': True})
                    html_section += f"""
                    <details>
                        <summary>{html.escape(title)}</summary>
                        <div class="chart-container">
                            {chart_html}
                        </div>
                    </details>
                    """
        except Exception as e:
            html_section += f"<p>Error generating visualizations: {html.escape(str(e))}</p>"
        return html_section

    def _generate_technical_indicators(self, tech_indicators) -> str:
        """Generate technical indicators section."""
        html_section = "<p>Trading signals based on technical indicators.</p>"
        try:
            signals = tech_indicators.get_trading_signals() if hasattr(tech_indicators, 'get_trading_signals') else {}
            if signals:
                html_section += """
                <table>
                    <tr><th>Indicator</th><th>Signal</th><th>Strength</th><th>Explanation</th></tr>
                """
                for indicator, signal_data in signals.items():
                    signal = signal_data.get('signal', 'Unknown')
                    strength = signal_data.get('strength', 'Moderate')
                    explanation = ""
                    if indicator == "RSI" and hasattr(tech_indicators, 'calculate_rsi'):
                        rsi_val = tech_indicators.calculate_rsi().iloc[-1] if not tech_indicators.calculate_rsi().empty else 0
                        explanation = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else f"Neutral (RSI={rsi_val:.1f})"
                    elif indicator == "MACD":
                        explanation = "Momentum via MACD crossovers."
                    elif indicator == "Bollinger Bands":
                        explanation = "Volatility and reversal signals."
                    signal_class = 'signal-buy' if 'buy' in signal.lower() else 'signal-sell' if 'sell' in signal.lower() else 'signal-hold'
                    html_section += f"""
                    <tr>
                        <td>{html.escape(indicator)}</td>
                        <td class="{signal_class}">{html.escape(signal)}</td>
                        <td>{html.escape(strength)}</td>
                        <td>{html.escape(explanation)}</td>
                    </tr>
                    """
                html_section += "</table>"
            else:
                html_section += "<p>No trading signals available.</p>"
        except Exception as e:
            html_section += f"<p>Error generating technical indicators: {html.escape(str(e))}</p>"
        return html_section

    def _generate_trading_insights(self, tech_indicators, analytics) -> str:
        """Generate trading insights section."""
        html_section = ""
        try:
            trading_signals = tech_indicators.get_trading_signals() if hasattr(tech_indicators, 'get_trading_signals') else {}
            signal_summary = {"buy": 0, "sell": 0, "hold": 0}
            for _, signal_data in trading_signals.items():
                signal = signal_data.get('signal', '').lower()
                if 'buy' in signal:
                    signal_summary["buy"] += 1
                elif 'sell' in signal:
                    signal_summary["sell"] += 1
                else:
                    signal_summary["hold"] += 1
            
            html_section += f"""
            <h3>Signal Summary</h3>
            <div class="metric-card">
                <div class="metric"><strong>Buy Signals</strong><br>{signal_summary['buy']}</div>
                <div class="metric"><strong>Sell Signals</strong><br>{signal_summary['sell']}</div>
                <div class="metric"><strong>Hold/Neutral</strong><br>{signal_summary['hold']}</div>
            </div>
            """
            
            html_section += "<h3>Overall Recommendation</h3>"
            if signal_summary["buy"] > signal_summary["sell"]:
                html_section += "<p class='signal-buy'>Bullish Outlook: Consider accumulating or holding positions.</p>"
            elif signal_summary["sell"] > signal_summary["buy"]:
                html_section += "<p class='signal-sell'>Bearish Outlook: Consider reducing positions.</p>"
            else:
                html_section += "<p class='signal-hold'>Neutral Outlook: Wait for clearer signals.</p>"
            
            strategies = analytics.generate_trading_strategies(trading_signals) if hasattr(analytics, 'generate_trading_strategies') else []
            if strategies:
                html_section += """
                <details>
                    <summary>Recommended Trading Strategies</summary>
                    <div>
                """
                for i, strategy in enumerate(strategies):
                    html_section += f"""
                        <h4>Strategy {i+1}: {html.escape(strategy.get('name', 'Unknown'))}</h4>
                        <p><strong>Type:</strong> {html.escape(strategy.get('type', 'N/A'))}</p>
                        <p><strong>Risk Level:</strong> {html.escape(strategy.get('risk_level', 'N/A'))}</p>
                        <p><strong>Time Horizon:</strong> {html.escape(strategy.get('time_horizon', 'N/A'))}</p>
                        <p><strong>Description:</strong> {html.escape(strategy.get('description', 'N/A'))}</p>
                    """
                html_section += "</div></details>"
            
            risk_metrics = analytics.calculate_risk_metrics() if hasattr(analytics, 'calculate_risk_metrics') else {}
            if risk_metrics:
                html_section += """
                <details>
                    <summary>Risk Metrics</summary>
                    <div class="metric-card">
                """
                for metric, value in risk_metrics.items():
                    value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    html_section += f'<div class="metric"><strong>{html.escape(metric.replace("_", " ").title())}:</strong><br>{html.escape(value_str)}</div>'
                html_section += "</div></details>"
            
            patterns = analytics.analyze_patterns() if hasattr(analytics, 'analyze_patterns') else {}
            if patterns:
                html_section += """
                <details>
                    <summary>Market Patterns Analysis</summary>
                    <div>
                """
                if patterns.get('seasonal_patterns'):
                    html_section += "<h4>Seasonal Patterns</h4><ul>"
                    for pattern in patterns['seasonal_patterns']:
                        html_section += f"<li>{html.escape(pattern)}</li>"
                    html_section += "</ul>"
                else:
                    html_section += "<p>No significant seasonal patterns detected.</p>"
                if patterns.get('volume_patterns'):
                    html_section += "<h4>Volume Patterns</h4><ul>"
                    for pattern in patterns['volume_patterns']:
                        html_section += f"<li>{html.escape(pattern)}</li>"
                    html_section += "</ul>"
                else:
                    html_section += "<p>No significant volume patterns detected.</p>"
                html_section += "</div></details>"
        except Exception as e:
            html_section += f"<p>Error generating trading insights: {html.escape(str(e))}</p>"
        return html_section

    def _generate_price_predictions(self, historical_data, predictions) -> str:
        """Generate price predictions section with optimized chart size."""
        html_section = ""
        try:
            if len(historical_data) > 50:
                pred_days = 7
                prediction_methods = ["technical_analysis", "moving_average", "learning_trend"]
                # Aggressive downsampling to reduce size
                recent_data = historical_data.tail(15).iloc[::3]  # Take every third point
                historical_dates = recent_data.index
                historical_prices = recent_data['Close'].values if 'Close' in recent_data.columns else np.zeros(len(recent_data))
                
                for method in prediction_methods:
                    pred_prices = predictions.predict_prices(pred_days, method=method) if hasattr(predictions, 'predict_prices') else []
                    if not pred_prices or not isinstance(pred_prices, (list, np.ndarray)) or len(pred_prices) != pred_days:
                        pred_prices = [historical_data['Close'].iloc[-1]] * pred_days
                        print(f"Warning: Invalid or empty prediction data for {method}, using fallback prices")
                    
                    current_price = historical_data['Close'].iloc[-1] if 'Close' in historical_data.columns else 0
                    predicted_final = pred_prices[-1] if pred_prices else current_price
                    change_pct = ((predicted_final - current_price) / current_price) * 100 if current_price != 0 else 0
                    confidence = predictions.calculate_prediction_confidence() if hasattr(predictions, 'calculate_prediction_confidence') else {}
                    
                    future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=pred_days, freq='D') if pd.api.types.is_datetime64_any_dtype(historical_dates) else pd.date_range(start=pd.Timestamp('2025-06-15') + pd.Timedelta(days=1), periods=pred_days, freq='D')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices', line=dict(color='#2b6cb0')))
                    fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, mode='lines+markers', name=f'Predicted Prices ({method.replace("_", " ").title()})', line=dict(color='#48bb78' if method == "technical_analysis" else '#f56565' if method == "moving_average" else '#ecc94b', dash='dash')))
                    fig.update_layout(title=f'7-Day Price Prediction ({method.replace("_", " ").title()})', xaxis_title='Date', yaxis_title='Price ($)', hovermode='closest', height=300, width=500, margin=dict(l=30, r=30, t=30, b=30))
                    chart_html = pio.to_html(fig, full_html=False, config={'displayModeBar': False, 'responsive': True})
                    
                    html_section += f"""
                    <details>
                        <summary>Price Prediction: {method.replace("_", " ").title()}</summary>
                        <div>
                            <div class="chart-container">
                                {chart_html}
                            </div>
                            <h3>Predicted Prices Table</h3>
                            <table>
                                <tr><th>Date</th><th>Predicted Price ($)</th></tr>
                            """
                    for date, price in zip(future_dates, pred_prices):
                        html_section += f"<tr><td>{date.strftime('%Y-%m-%d')}</td><td>{float(price):.2f}</td></tr>"
                    html_section += "</table>"
                    
                    html_section += f"""
                            <div class="metric-card">
                                <div class="metric"><strong>Predicted Change</strong><br>{change_pct:.2f}%</div>
                                <div class="metric"><strong>Target Price</strong><br>${predicted_final:.2f}</div>
                                <div class="metric"><strong>Confidence Score</strong><br>{confidence.get('score', 0):.1f}/10</div>
                                <div class="metric"><strong>Prediction Volatility</strong><br>{confidence.get('volatility', 0):.2f}%</div>
                            </div>
                            <p><em>Disclaimer:</em> Predictions are for informational purposes only.</p>
                        </div>
                    </details>
                    """
            else:
                html_section += "<p>Insufficient data for predictions (minimum 50 data points).</p>"
        except Exception as e:
            html_section += f"<p>Error generating predictions: {html.escape(str(e))}</p>"
        return html_section

    def _generate_comparative_analysis(self, advanced_analytics) -> str:
        """Generate comparative analysis section."""
        html_section = ""
        try:
            summary = advanced_analytics.get_performance_summary() if hasattr(advanced_analytics, 'get_performance_summary') else {}
            if summary:
                html_section += f"""
                <div class="metric-card">
                    <div class="metric"><strong>Total Stocks</strong><br>{summary.get('total_stocks', 0)}</div>
                    <div class="metric"><strong>Average Change</strong><br>{summary.get('avg_change', 0):.2f}%</div>
                    <div class="metric"><strong>Gainers</strong><br>{summary.get('gainers', 0)}</div>
                    <div class="metric"><strong>Losers</strong><br>{summary.get('losers', 0)}</div>
                </div>
                """
            
            sector_analysis = advanced_analytics.get_sector_analysis() if hasattr(advanced_analytics, 'get_sector_analysis') else pd.DataFrame()
            if not sector_analysis.empty:
                html_section += """
                <details>
                    <summary>Sector Performance</summary>
                    <div>
                        <table>
                """
                html_section += sector_analysis.to_html(index=False, border=1, classes="table", escape=False)
                html_section += "</table></div></details>"
        except Exception as e:
            html_section += f"<p>Error generating comparative analysis: {html.escape(str(e))}</p>"
        return html_section

    def _generate_data_summary(self, data: pd.DataFrame) -> str:
        """Generate data summary section."""
        html_section = ""
        try:
            if not data.empty:
                summary_stats = data['Close'].describe()
                html_section += """
                <details>
                    <summary>Data Summary</summary>
                    <div>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                """
                for stat, value in summary_stats.items():
                    html_section += f"<tr><td>{html.escape(stat.capitalize())}</td><td>{value:.2f}</td></tr>"
                html_section += "</table></div></details>"
            else:
                html_section += "<p>No data available for summary.</p>"
        except Exception as e:
            html_section += f"<p>Error generating data summary: {html.escape(str(e))}</p>"
        return html_section

    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content.decode('utf-8'))
        return filename
