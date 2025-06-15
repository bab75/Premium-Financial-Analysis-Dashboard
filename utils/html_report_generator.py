"""
HTML Report Generator for Financial Analysis
Creates comprehensive downloadable HTML reports with interactive charts
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz
from typing import Dict, List, Optional
import base64
from io import StringIO
import html
import logging

logging.basicConfig(level=logging.INFO)

class HTMLReportGenerator:
    """Generate comprehensive HTML reports with interactive charts and analysis."""
    
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
                
        except Exception as e:
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
        <style>
            body {
                font-family: 'Roboto', Arial, sans-serif;
                margin: 20px;
                background-color: #f5f7fa;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            h1 {
                color: #1a3c6e;
                text-align: center;
                font-size: 2.2em;
                margin-bottom: 10px;
                background: linear-gradient(90deg, #1a3c6e, #3b6e9c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            h2 {
                color: #2a5d9a;
                font-size: 1.8em;
                margin: 20px 0 10px;
            }
            h3 {
                color: #3b6e9c;
                font-size: 1.4em;
            }
            .tab {
                overflow: hidden;
                background: #e9ecef;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .tab button {
                background: linear-gradient(90deg, #e9ecef, #f8f9fa);
                border: none;
                outline: none;
                cursor: pointer;
                padding: 12px 20px;
                transition: all 0.3s ease;
                font-size: 1em;
                color: #2a5d9a;
                float: left;
                border-radius: 8px 8px 0 0;
            }
            .tab button:hover {
                background: linear-gradient(90deg, #d6d9dc, #e9ecef);
                transform: translateY(-2px);
            }
            .tab button.active {
                background: white;
                color: #1a3c6e;
                box-shadow: inset 0 -2px 0 #1a3c6e;
            }
            .tabcontent {
                display: none;
                padding: 20px;
                background: #fff;
                border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                animation: fadeIn 0.5s ease-in;
            }
            .tabcontent.active {
                display: block;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            details {
                margin: 15px 0;
                background: #f8f9fa;
                border-radius: 6px;
                padding: 10px;
                transition: all 0.3s ease;
            }
            summary {
                cursor: pointer;
                font-weight: bold;
                color: #2a5d9a;
                padding: 10px;
                display: flex;
                align-items: center;
                transition: color 0.3s ease;
            }
            summary:hover {
                color: #1a3c6e;
            }
            summary::marker {
                content: "▶ ";
                color: #2a5d9a;
            }
            details[open] summary::marker {
                content: "▼ ";
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
                background: #fff;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #e9ecef;
                padding: 12px;
                text-align: left;
            }
            th {
                background: linear-gradient(90deg, #2a5d9a, #3b6e9c);
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            tr:hover {
                background-color: #e9ecef;
            }
            .signal-buy {
                color: #28a745;
                font-weight: bold;
            }
            .signal-sell {
                color: #dc3545;
                font-weight: bold;
            }
            .signal-hold {
                color: #007bff;
                font-weight: bold;
            }
            .error {
                color: #dc3545;
                font-style: italic;
            }
            .risk-high { color: #dc3545; }
            .risk-medium { color: #ffc107; }
            .risk-low { color: #28a745; }
            .plotly-chart {
                margin: 20px 0;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }
            @media screen and (max-width: 768px) {
                .tab button {
                    display: block;
                    width: 100%;
                    border-radius: 0;
                }
                table {
                    font-size: 14px;
                    display: block;
                    overflow-x: auto;
                }
                .container {
                    padding: 10px;
                }
            }
        </style>
        """
        self.js_script = """
        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            document.addEventListener("DOMContentLoaded", function() {
                document.getElementsByClassName("tablinks")[0].click();
            });
        </script>
        """
    
    def generate_comprehensive_report(self, 
                                    stock_symbol: str,
                                    historical_data: pd.DataFrame,
                                    tech_indicators,
                                    analytics,
                                    visualizations,
                                    predictions=None,
                                    advanced_analytics=None,
                                    report_type: str = "full") -> str:
        """Generate a comprehensive HTML report with all analysis components."""
        logging.info(f"Generating report for {stock_symbol}, report_type: {report_type}")
        
        historical_data = self._clean_dataframe(historical_data)
        
        local_tz = datetime.now().astimezone().tzinfo
        timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        stock_symbol = html.escape(str(stock_symbol))
        timestamp = html.escape(str(timestamp))
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Financial Analysis Report - {stock_symbol}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    {self.css_styles}
</head>
<body>
    <div class="container">
        <h1>Financial Analysis Report</h1>
        <h2>Stock Symbol: {stock_symbol} | Generated: {timestamp}</h2>
        
        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'ExecutiveSummary')">Executive Summary</button>
            <button class="tablinks" onclick="openTab(event, 'TechnicalAnalysis')">Technical Analysis</button>
            <button class="tablinks" onclick="openTab(event, 'TradingSignals')">Trading Signals</button>
            <button class="tablinks" onclick="openTab(event, 'PriceCharts')">Price Charts</button>
            <button class="tablinks" onclick="openTab(event, 'PerformanceMetrics')">Performance Metrics</button>
            <button class="tablinks" onclick="openTab(event, 'RiskAnalysis')">Risk Analysis</button>
"""
        if predictions is not None and report_type in ["full", "predictions"]:
            html_content += """
            <button class="tablinks" onclick="openTab(event, 'PricePredictions')">Price Predictions</button>
"""
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += """
            <button class="tablinks" onclick="openTab(event, '3DVisualizations')">3D Visualizations</button>
"""
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += """
            <button class="tablinks" onclick="openTab(event, 'AdvancedAnalytics')">Advanced Analytics</button>
"""
        html_content += """
        </div>
        
        <div id="ExecutiveSummary" class="tabcontent">
            {self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)}
        </div>
        <div id="TechnicalAnalysis" class="tabcontent">
            {self._generate_technical_charts_section(tech_indicators)}
        </div>
        <div id="TradingSignals" class="tabcontent">
            {self._generate_trading_signals_section(tech_indicators)}
        </div>
        <div id="PriceCharts" class="tabcontent">
            {self._generate_price_charts_section(visualizations)}
        </div>
        <div id="PerformanceMetrics" class="tabcontent">
            {self._generate_performance_metrics(historical_data)}
        </div>
        <div id="RiskAnalysis" class="tabcontent">
            {self._generate_risk_analysis(analytics, historical_data)}
        </div>
"""
        if predictions is not None and report_type in ["full", "predictions"]:
            html_content += f"""
        <div id="PricePredictions" class="tabcontent">
            {self._generate_prediction_charts_section(historical_data)}
        </div>
"""
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += f"""
        <div id="3DVisualizations" class="tabcontent">
            {self._generate_3d_charts_section(visualizations)}
        </div>
"""
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += f"""
        <div id="AdvancedAnalytics" class="tabcontent">
            {self._generate_advanced_analytics_section(advanced_analytics)}
        </div>
"""
        html_content += f"""
        <footer>
            <p>This report was generated automatically by the Financial Analysis Application</p>
            <p>Data analysis period: {str(historical_data.index[0])[:10]} to {str(historical_data.index[-1])[:10]}</p>
        </footer>
    </div>
    {self.js_script}
</body>
</html>
"""
        return html_content
    
    def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
        """Generate executive summary section."""
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
        
        high_52week = data['High'].max()
        low_52week = data['Low'].min()
        avg_volume = data['Volume'].mean()
        
        html_section = f"""
    <h2>📊 Executive Summary</h2>
    <table class="summary-table">
        <tr><th>Current Price</th><td>${current_price:.2f}</td></tr>
        <tr><th>Daily Change</th><td>{price_change:+.2f} ({price_change_pct:+.2f}%)</td></tr>
        <tr><th>52-Week High</th><td>${high_52week:.2f}</td></tr>
        <tr><th>52-Week Low</th><td>${low_52week:.2f}</td></tr>
        <tr><th>Average Volume</th><td>{avg_volume:,.0f}</td></tr>
    </table>
"""
        return html_section
    
    def _generate_technical_charts_section(self, tech_indicators) -> str:
        """Generate technical analysis charts section."""
        html_section = """
    <h2>📈 Technical Analysis Charts</h2>
    <p>Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.</p>
"""
        try:
            ma_chart = tech_indicators.create_moving_averages_chart()
            html_section += f"""
    <details>
        <summary>Moving Averages</summary>
        <div class="plotly-chart">{ma_chart.to_html(include_plotlyjs='inline', div_id='ma-chart')}</div>
    </details>
"""
            rsi_chart = tech_indicators.create_rsi_chart()
            html_section += f"""
    <details>
        <summary>Relative Strength Index (RSI)</summary>
        <div class="plotly-chart">{rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')}</div>
    </details>
"""
            macd_chart = tech_indicators.create_macd_chart()
            html_section += f"""
    <details>
        <summary>MACD Analysis</summary>
        <div class="plotly-chart">{macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')}</div>
    </details>
"""
            bb_chart = tech_indicators.create_bollinger_bands_chart()
            html_section += f"""
    <details>
        <summary>Bollinger Bands</summary>
        <div class="plotly-chart">{bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')}</div>
    </details>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error generating technical charts: {html.escape(str(e))}</p>
"""
        return html_section
    
    def _generate_trading_signals_section(self, tech_indicators) -> str:
        """Generate trading signals analysis section."""
        html_section = """
    <h2>🎯 Trading Signals & Recommendations</h2>
"""
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
                html_section += f"""
    <div>
        <h3>{html.escape(indicator)}</h3>
        <p><span class="{signal_class}">Signal: {html.escape(signal_data.get('signal', 'Unknown'))}</span></p>
        <p>Strength: {html.escape(strength)}</p>
    </div>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error generating trading signals: {html.escape(str(e))}</p>
"""
        return html_section
    
    def _generate_price_charts_section(self, visualizations) -> str:
        """Generate price visualization charts section."""
        html_section = """
    <h2>💹 Price Analysis Charts</h2>
"""
        try:
            candlestick_chart = visualizations.create_candlestick_chart()
            html_section += f"""
    <details>
        <summary>Candlestick Chart</summary>
        <div class="plotly-chart">{candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')}</div>
    </details>
"""
            trends_chart = visualizations.create_price_trends_chart()
            html_section += f"""
    <details>
        <summary>Price Trends</summary>
        <div class="plotly-chart">{trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')}</div>
    </details>
"""
            volume_chart = visualizations.create_volume_chart()
            html_section += f"""
    <details>
        <summary>Volume Analysis</summary>
        <div class="plotly-chart">{volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')}</div>
    </details>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error generating price charts: {html.escape(str(e))}</p>
"""
        return html_section
    
    def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
        """Generate performance metrics table."""
        html_section = """
    <h2>📊 Performance Metrics</h2>
    <table class="summary-table">
        <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
"""
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
                html_section += f"""
        <tr>
            <td>{html.escape(metric)}</td>
            <td>{html.escape(value)}</td>
            <td>{html.escape(description)}</td>
        </tr>
"""
        except Exception as e:
            html_section += f"""
        <tr><td colspan="3" class="error">Error calculating metrics: {html.escape(str(e))}</td></tr>
"""
        html_section += """
    </table>
"""
        return html_section
    
    def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        html_section = """
    <h2>⚠️ Risk Analysis</h2>
    <table class="summary-table">
"""
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
            
            html_section += f"""
        <tr><th>Annualized Volatility</th><td>{volatility:.2f}%</td></tr>
        <tr><th>Maximum Drawdown</th><td>{max_drawdown:.2f}%</td></tr>
"""
            html_section += """
    </table>
    <h3>Risk Assessment</h3>
"""
            if volatility > 30:
                html_section += """
    <p class="risk-high">🔴 High Risk: This stock shows high volatility. Suitable for experienced traders with high risk tolerance.</p>
"""
            elif volatility > 20:
                html_section += """
    <p class="risk-medium">🟡 Medium Risk: Moderate volatility. Suitable for balanced investment strategies.</p>
"""
            else:
                html_section += """
    <p class="risk-low">🟢 Low Risk: Relatively stable price movements. Suitable for conservative investors.</p>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error calculating risk metrics: {html.escape(str(e))}</p>
"""
        return html_section
    
    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_report_{timestamp}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return filename
    
    def _generate_prediction_charts_section(self, historical_data) -> str:
        """Generate prediction charts and table section for HTML report."""
        html_section = """
    <h2>📈 Price Predictions</h2>
    <p>Technical analysis, linear trend, and moving average predictions for future price movements.</p>
"""
        try:
            from utils.predictions import PricePredictions
            predictions = PricePredictions(historical_data)
            prediction_days = 7
            methods = [
                ("technical_analysis", "Technical Analysis Prediction", "Advanced technical indicators and momentum analysis"),
                ("linear_trend", "Linear Trend Prediction", "Statistical trend analysis and regression modeling"),
                ("moving_average", "Moving Average Prediction", "Simple and exponential moving average forecasting")
            ]
            
            prediction_table_data = []
            future_dates = None
            
            for method_key, method_name, method_desc in methods:
                try:
                    pred_data = predictions.predict_prices(prediction_days, method_key)
                    if pred_data and len(pred_data) > 0:
                        recent_data = historical_data.tail(20)
                        historical_dates = recent_data.index if hasattr(recent_data, 'index') else range(len(recent_data))
                        historical_prices = recent_data['Close'].values
                        
                        import pandas as pd
                        if isinstance(historical_dates, pd.DatetimeIndex) and len(historical_dates) > 0:
                            last_date = historical_dates[-1]
                        else:
                            last_date = pd.Timestamp.now()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
                        
                        for date, price in zip(future_dates, pred_data):
                            prediction_table_data.append({
                                'Date': date.strftime('%Y-%m-%d'),
                                'Method': method_name,
                                'Predicted Price': f"${price:.2f}"
                            })
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=historical_dates,
                            y=historical_prices,
                            mode='lines',
                            name='Historical Prices',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=pred_data,
                            mode='lines+markers',
                            name=method_name,
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            title=f"{method_name} - {prediction_days} Day Forecast",
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified',
                            showlegend=True,
                            height=400
                        )
                        chart_html = fig.to_html(include_plotlyjs=False, div_id=f"prediction_{method_key}")
                        
                        html_section += f"""
    <details>
        <summary>{html.escape(method_name)}</summary>
        <p>{html.escape(method_desc)}</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                except Exception as method_error:
                    html_section += f"""
    <p class="error">Error generating {html.escape(method_name)}: {html.escape(str(method_error))}</p>
"""
            
            if prediction_table_data:
                html_section += """
    <details>
        <summary>Price Prediction Summary Table</summary>
        <table class="summary-table">
            <tr><th>Date</th><th>Method</th><th>Predicted Price</th></tr>
"""
                for row in prediction_table_data:
                    html_section += f"""
            <tr>
                <td>{html.escape(row['Date'])}</td>
                <td>{html.escape(row['Method'])}</td>
                <td>{html.escape(row['Predicted Price'])}</td>
            </tr>
"""
                html_section += """
        </table>
    </details>
"""
            
            try:
                confidence_data = predictions.calculate_prediction_confidence()
                disclaimer = predictions.get_prediction_disclaimer()
                metrics = [
                    ("Confidence Level", confidence_data.get('confidence_level', 'N/A'), "%"),
                    ("Trend Strength", confidence_data.get('trend_strength', 'N/A'), ""),
                    ("Data Quality", confidence_data.get('data_quality', 'N/A'), ""),
                    ("Volatility Risk", confidence_data.get('volatility_risk', 0), "%")
                ]
                
                html_section += """
    <details>
        <summary>Prediction Metrics & Reliability</summary>
        <table class="summary-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
                for metric, value, suffix in metrics:
                    formatted_value = f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else str(value)
                    html_section += f"""
            <tr>
                <td>{html.escape(metric)}</td>
                <td>{html.escape(formatted_value)}</td>
            </tr>
"""
                html_section += """
        </table>
        <p>{html.escape(disclaimer)}</p>
    </details>
"""
            except Exception as metrics_error:
                html_section += f"""
    <p class="error">Error generating prediction metrics: {html.escape(str(metrics_error))}</p>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error initializing predictions: {html.escape(str(e))}</p>
"""
        return html_section
    
    def _generate_3d_charts_section(self, visualizations) -> str:
        """Generate 3D visualization charts section for HTML report."""
        html_section = """
    <h2>📊 3D Visualizations</h2>
    <p>Interactive three-dimensional analysis for comprehensive market insights.</p>
"""
        content_added = False
        try:
            logging.info(f"Processing 3D visualizations: {visualizations is not None}")
            if hasattr(visualizations, 'get_3d_price_volume_chart'):
                chart = visualizations.get_3d_price_volume_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_price_volume")
                    html_section += f"""
    <details>
        <summary>3D Price-Volume Analysis</summary>
        <p>Three-dimensional visualization of price movements, volume, and time relationships.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if hasattr(visualizations, 'get_3d_technical_surface'):
                chart = visualizations.get_3d_technical_surface()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_technical_surface")
                    html_section += f"""
    <details>
        <summary>3D Technical Indicator Surface</summary>
        <p>Surface plot showing relationships between multiple technical indicators.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if hasattr(visualizations, 'get_3d_market_dynamics'):
                chart = visualizations.get_3d_market_dynamics()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_market_dynamics")
                    html_section += f"""
    <details>
        <summary>3D Market Dynamics</summary>
        <p>Multi-dimensional view of market behavior and trading patterns.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if not content_added:
                html_section += """
    <p>No 3D visualizations available. Ensure visualization methods return valid charts.</p>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error generating 3D charts: {html.escape(str(e))}</p>
"""
        return html_section
    
    def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
        """Generate advanced analytics section for HTML report."""
        html_section = """
    <h2>🎯 Advanced Analytics</h2>
    <p>Comprehensive sector analysis, correlations, and market intelligence.</p>
"""
        content_added = False
        try:
            logging.info(f"Processing advanced analytics: {advanced_analytics is not None}")
            if hasattr(advanced_analytics, 'create_sector_performance_chart'):
                chart = advanced_analytics.create_sector_performance_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="sector_performance")
                    html_section += f"""
    <details>
        <summary>Sector Performance Analysis</summary>
        <p>Comparative performance across different market sectors.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if hasattr(advanced_analytics, 'create_correlation_heatmap'):
                chart = advanced_analytics.create_correlation_heatmap()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="correlation_heatmap")
                    html_section += f"""
    <details>
        <summary>Market Correlation Analysis</summary>
        <p>Heat map showing correlations between different market metrics and indicators.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if hasattr(advanced_analytics, 'create_performance_dashboard'):
                chart = advanced_analytics.create_performance_dashboard()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="performance_dashboard")
                    html_section += f"""
    <details>
        <summary>Comprehensive Performance Dashboard</summary>
        <p>Multi-metric dashboard showing key performance indicators and trends.</p>
        <div class="plotly-chart">{chart_html}</div>
    </details>
"""
                    content_added = True
            if hasattr(advanced_analytics, 'get_industry_analysis'):
                industry_data = advanced_analytics.get_industry_analysis()
                if industry_data is not None and not industry_data.empty:
                    html_section += f"""
    <details>
        <summary>Industry Analysis Summary</summary>
        <table class="summary-table">{industry_data.head(10).to_html(classes="summary-table", escape=False, border=0)}</table>
    </details>
"""
                    content_added = True
            if not content_added:
                html_section += """
    <p>No advanced analytics data available.</p>
"""
        except Exception as e:
            html_section += f"""
    <p class="error">Error generating advanced analytics: {html.escape(str(e))}</p>
"""
        return html_section
