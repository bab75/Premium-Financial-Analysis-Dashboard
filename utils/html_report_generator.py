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
        if data is None or data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        try:
            if 'Datetime' in data.columns:
                if isinstance(data.index, pd.DatetimeIndex) or (
                    hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype).lower()
                ):
                    data = data.reset_index().rename(columns={'index': 'Datetime'}).drop(columns=['Datetime'], errors='ignore')
                else:
                    data = data.reset_index().rename(columns={'index': 'Datetime'})
            if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.set_index('Date').rename_axis('Datetime')
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index(drop=True)
                if 'Datetime' not in data.columns:
                    data['Datetime'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
            data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
            data = data.dropna(subset=['Datetime'])
            return data.set_index('Datetime')
        except Exception as e:
            logging.error(f"Data cleaning failed: {str(e)}")
            return data.reset_index(drop=True)

    def __init__(self):
        self.css_styles = """
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
                h1 { color: #2c3e50; text-align: center; }
                h2, h3 { color: #34495e; }
                .chart-container { margin: 20px 0; padding: 10px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .section { margin: 30px 0; }
                .metric-card { display: flex; flex-wrap: wrap; gap: 10px; }
                .metric { background-color: #e8ecef; padding: 10px; border-radius: 5px; flex: 1; min-width: 150px; text-align: center; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .disclaimer { font-style: italic; color: #7f8c8d; font-size: 0.9em; }
                .signal-buy { color: #27ae60; font-weight: bold; }
                .signal-sell { color: #c0392b; font-weight: bold; }
                .signal-hold { color: #7f8c8d; font-weight: bold; }
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
                <h2>Executive Summary</h2>
                {self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)}
            </div>
            <div class="section">
                <h2>Technical Analysis Charts</h2>
                {self._generate_technical_charts_section(tech_indicators)}
            </div>
            <div class="section">
                <h2>Trading Signals & Recommendations</h2>
                {self._generate_trading_signals_section(tech_indicators)}
            </div>
            <div class="section">
                <h2>Price Analysis Charts</h2>
                {self._generate_price_charts_section(visualizations)}
            </div>
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._generate_performance_metrics(historical_data)}
            </div>
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._generate_risk_analysis(analytics, historical_data)}
            </div>
        """
        
        if predictions is not None and report_type in ["full", "predictions"]:
            html_content += f"""
            <div class="section">
                <h2>Price Predictions</h2>
                {self._generate_prediction_charts_section(historical_data)}
            </div>
            """
        
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += f"""
            <div class="section">
                <h2>3D Visualizations</h2>
                {self._generate_3d_charts_section(visualizations)}
            </div>
            """
        
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += f"""
            <div class="section">
                <h2>Advanced Analytics</h2>
                {self._generate_advanced_analytics_section(advanced_analytics)}
            </div>
            """
        
        html_content += f"""
            <div class="section disclaimer">
                <p><strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute financial advice. Always conduct your own research before making investment decisions.</p>
                <p>Data analysis period: {str(historical_data.index[0])[:10] if not historical_data.empty else 'N/A'} to {str(historical_data.index[-1])[:10] if not historical_data.empty else 'N/A'}</p>
                <p>This report was generated automatically by the Financial Analysis Application.</p>
            </div>
        </body>
        </html>
        """
        
        return html_content.encode('utf-8')

    def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
        """Generate executive summary section."""
        current_price = data['Close'].iloc[-1] if not data.empty else 0
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
        
        high_52week = data['High'].max() if not data.empty else 0
        low_52week = data['Low'].min() if not data.empty else 0
        avg_volume = data['Volume'].mean() if 'Volume' in data.columns else 0
        
        html_section = f"""
            <div class="metric-card">
                <div class="metric"><strong>Current Price</strong><br>${current_price:.2f}</div>
                <div class="metric"><strong>Daily Change</strong><br>{price_change:+.2f} ({price_change_pct:+.2f}%)</div>
                <div class="metric"><strong>52-Week High</strong><br>${high_52week:.2f}</div>
                <div class="metric"><strong>52-Week Low</strong><br>${low_52week:.2f}</div>
                <div class="metric"><strong>Average Volume</strong><br>{avg_volume:,.0f}</div>
            </div>
        """
        return html_section

    def _generate_technical_charts_section(self, tech_indicators) -> str:
        """Generate technical analysis charts section."""
        html_section = """
            <p>Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.</p>
        """
        try:
            ma_chart = tech_indicators.create_moving_averages_chart()
            rsi_chart = tech_indicators.create_rsi_chart()
            macd_chart = tech_indicators.create_macd_chart()
            bb_chart = tech_indicators.create_bollinger_bands_chart()
            
            html_section += """
            <div class="chart-container">
                <h3>Moving Averages</h3>
                {ma_chart.to_html(include_plotlyjs=False, div_id='ma-chart')}
            </div>
            <div class="chart-container">
                <h3>Relative Strength Index (RSI)</h3>
                {rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')}
            </div>
            <div class="chart-container">
                <h3>MACD Analysis</h3>
                {macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')}
            </div>
            <div class="chart-container">
                <h3>Bollinger Bands</h3>
                {bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')}
            </div>
            """
        except Exception as e:
            html_section += f"<p>Error generating technical charts: {html.escape(str(e))}</p>"
        return html_section.format(ma_chart=ma_chart, rsi_chart=rsi_chart, macd_chart=macd_chart, bb_chart=bb_chart)

    def _generate_trading_signals_section(self, tech_indicators) -> str:
        """Generate trading signals analysis section."""
        html_section = """
            <p>Trading signals and recommendations based on technical indicators.</p>
            <table>
                <tr><th>Indicator</th><th>Signal</th><th>Strength</th></tr>
        """
        try:
            signals = tech_indicators.get_trading_signals()
            for indicator, signal_data in signals.items():
                signal = signal_data.get('signal', 'Unknown')
                strength = signal_data.get('strength', 'Moderate')
                signal_class = 'signal-buy' if 'buy' in signal.lower() else 'signal-sell' if 'sell' in signal.lower() else 'signal-hold'
                html_section += f"""
                <tr>
                    <td>{html.escape(indicator)}</td>
                    <td class="{signal_class}">{html.escape(signal)}</td>
                    <td>{html.escape(strength)}</td>
                </tr>
                """
            html_section += "</table>"
        except Exception as e:
            html_section += f"<p>Error generating trading signals: {html.escape(str(e))}</p>"
        return html_section

    def _generate_price_charts_section(self, visualizations) -> str:
        """Generate price visualization charts section."""
        html_section = """
            <p>Interactive price analysis charts.</p>
        """
        try:
            candlestick_chart = visualizations.create_candlestick_chart()
            trends_chart = visualizations.create_price_trends_chart()
            volume_chart = visualizations.create_volume_chart()
            
            html_section += """
            <div class="chart-container">
                <h3>Candlestick Chart</h3>
                {candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')}
            </div>
            <div class="chart-container">
                <h3>Price Trends</h3>
                {trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')}
            </div>
            <div class="chart-container">
                <h3>Volume Analysis</h3>
                {volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')}
            </div>
            """
        except Exception as e:
            html_section += f"<p>Error generating price charts: {html.escape(str(e))}</p>"
        return html_section.format(candlestick_chart=candlestick_chart, trends_chart=trends_chart, volume_chart=volume_chart)

    def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
        """Generate performance metrics table."""
        html_section = """
            <table>
                <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
        """
        try:
            total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100 if not data.empty and len(data) > 1 else 0
            volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100 if not data.empty else 0
            max_price = data['High'].max() if not data.empty else 0
            min_price = data['Low'].min() if not data.empty else 0
            avg_volume = data['Volume'].mean() if 'Volume' in data.columns else 0
            
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
                <tr><td>{html.escape(metric)}</td><td>{html.escape(value)}</td><td>{html.escape(description)}</td></tr>
                """
            html_section += "</table>"
        except Exception as e:
            html_section += f"<tr><td colspan='3'>Error calculating metrics: {html.escape(str(e))}</td></tr></table>"
        return html_section

    def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        html_section = """
            <div>
                <h3>Risk Metrics</h3>
        """
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100 if not returns.empty else 0
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100 if not data.empty else 0
            
            html_section += f"""
                <div class="metric-card">
                    <div class="metric"><strong>Annualized Volatility</strong><br>{volatility:.2f}%</div>
                    <div class="metric"><strong>Maximum Drawdown</strong><br>{max_drawdown:.2f}%</div>
                </div>
                <h3>Risk Assessment</h3>
            """
            if volatility > 30:
                html_section += "<p class='signal-sell'>🔴 High Risk: This stock shows high volatility. Suitable for experienced traders with high risk tolerance.</p>"
            elif volatility > 20:
                html_section += "<p class='signal-hold'>🟡 Medium Risk: Moderate volatility. Suitable for balanced investment strategies.</p>"
            else:
                html_section += "<p class='signal-buy'>🟢 Low Risk: Relatively stable price movements. Suitable for conservative investors.</p>"
        except Exception as e:
            html_section += f"<p>Error calculating risk metrics: {html.escape(str(e))}</p>"
        html_section += "</div>"
        return html_section

    def _generate_prediction_charts_section(self, historical_data) -> str:
        """Generate prediction charts and table section for HTML report."""
        html_section = """
            <p>Technical analysis, linear trend, and moving average predictions for future price movements.</p>
        """
        try:
            from utils.price_predictions import PricePredictions
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
                        historical_dates = recent_data.index if hasattr(recent_data, 'index') and not recent_data.index.empty else pd.date_range(start='2020-01-01', periods=20, freq='D')
                        historical_prices = recent_data['Close'].values if not recent_data.empty else np.zeros(20)
                        
                        if isinstance(historical_dates, pd.DatetimeIndex) and len(historical_dates) > 0:
                            last_date = historical_dates[-1]
                        else:
                            last_date = pd.Timestamp.now()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices', line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=future_dates, y=pred_data, mode='lines+markers', name=method_name, line=dict(color='red', width=2, dash='dash'), marker=dict(size=6)))
                        fig.update_layout(title=f"{method_name} - {prediction_days} Day Forecast", xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified', showlegend=True, height=400)
                        chart_html = fig.to_html(include_plotlyjs=False, div_id=f"prediction_{method_key}")
                        
                        html_section += f"""
                <div class="chart-container">
                    <h3>{html.escape(method_name)}</h3>
                    <p>{html.escape(method_desc)}</p>
                    {chart_html}
                </div>
                """
                        for date, price in zip(future_dates, pred_data):
                            prediction_table_data.append({
                                'Date': date.strftime('%Y-%m-%d'),
                                'Method': method_name,
                                'Predicted Price': f"${price:.2f}"
                            })
                except Exception as method_error:
                    html_section += f"<p>Error generating {html.escape(method_name)}: {html.escape(str(method_error))}</p>"
            
            if prediction_table_data:
                html_section += """
                <h3>Price Prediction Summary Table</h3>
                <table>
                    <tr><th>Date</th><th>Method</th><th>Predicted Price</th></tr>
                """
                for row in prediction_table_data:
                    html_section += f"""
                    <tr><td>{html.escape(row['Date'])}</td><td>{html.escape(row['Method'])}</td><td>{html.escape(row['Predicted Price'])}</td></tr>
                    """
                html_section += "</table>"
            
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
                <h3>Prediction Metrics & Reliability</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                """
                for metric, value, suffix in metrics:
                    formatted_value = f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else str(value)
                    html_section += f"""
                    <tr><td>{html.escape(metric)}</td><td>{html.escape(formatted_value)}</td></tr>
                    """
                html_section += "</table>"
                html_section += f"<p>{html.escape(disclaimer)}</p>"
            except Exception as metrics_error:
                html_section += f"<p>Error generating prediction metrics: {html.escape(str(metrics_error))}</p>"
        except Exception as e:
            html_section += f"<p>Error initializing predictions: {html.escape(str(e))}</p>"
        return html_section

    def _generate_3d_charts_section(self, visualizations) -> str:
        """Generate 3D visualization charts section for HTML report."""
        html_section = """
            <p>Interactive three-dimensional analysis for comprehensive market insights.</p>
        """
        content_added = False
        try:
            if hasattr(visualizations, 'get_3d_price_volume_chart'):
                chart = visualizations.get_3d_price_volume_chart()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>3D Price-Volume Analysis</h3>
                        <p>Three-dimensional visualization of price movements, volume, and time relationships.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='3d_price_volume')}
                    </div>
                    """
                    content_added = True
            if hasattr(visualizations, 'get_3d_technical_surface'):
                chart = visualizations.get_3d_technical_surface()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>3D Technical Indicator Surface</h3>
                        <p>Surface plot showing relationships between multiple technical indicators.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='3d_technical_surface')}
                    </div>
                    """
                    content_added = True
            if hasattr(visualizations, 'get_3d_market_dynamics'):
                chart = visualizations.get_3d_market_dynamics()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>3D Market Dynamics</h3>
                        <p>Multi-dimensional view of market behavior and trading patterns.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='3d_market_dynamics')}
                    </div>
                    """
                    content_added = True
            if not content_added:
                html_section += "<p>No 3D visualizations available. Ensure visualization methods return valid charts.</p>"
        except Exception as e:
            html_section += f"<p>Error generating 3D charts: {html.escape(str(e))}</p>"
        return html_section.format(chart=chart) if content_added else html_section

    def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
        """Generate advanced analytics section for HTML report."""
        html_section = """
            <p>Comprehensive sector analysis, correlations, and market intelligence.</p>
        """
        content_added = False
        try:
            if hasattr(advanced_analytics, 'create_sector_performance_chart'):
                chart = advanced_analytics.create_sector_performance_chart()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>Sector Performance Analysis</h3>
                        <p>Comparative performance across different market sectors.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='sector_performance')}
                    </div>
                    """
                    content_added = True
            if hasattr(advanced_analytics, 'create_correlation_heatmap'):
                chart = advanced_analytics.create_correlation_heatmap()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>Market Correlation Analysis</h3>
                        <p>Heat map showing correlations between different market metrics and indicators.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='correlation_heatmap')}
                    </div>
                    """
                    content_added = True
            if hasattr(advanced_analytics, 'create_performance_dashboard'):
                chart = advanced_analytics.create_performance_dashboard()
                if chart:
                    html_section += """
                    <div class="chart-container">
                        <h3>Comprehensive Performance Dashboard</h3>
                        <p>Multi-metric dashboard showing key performance indicators and trends.</p>
                        {chart.to_html(include_plotlyjs=False, div_id='performance_dashboard')}
                    </div>
                    """
                    content_added = True
            if hasattr(advanced_analytics, 'get_industry_analysis'):
                industry_data = advanced_analytics.get_industry_analysis()
                if industry_data is not None and not industry_data.empty:
                    html_section += """
                    <div class="chart-container">
                        <h3>Industry Analysis Summary</h3>
                        {industry_data.head(10).to_html(classes='summary-table', escape=False)}
                    </div>
                    """
                    content_added = True
            if not content_added:
                html_section += "<p>No advanced analytics data available.</p>"
        except Exception as e:
            html_section += f"<p>Error generating advanced analytics: {html.escape(str(e))}</p>"
        return html_section.format(chart=chart) if content_added else html_section

    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_report_{timestamp}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content.decode('utf-8'))
        return filename
