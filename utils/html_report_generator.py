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
import html
import logging

logging.basicConfig(level=logging.INFO)

class HTMLReportGenerator:
    """Generate comprehensive HTML reports with interactive charts and analysis."""
    
    def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to resolve index/column ambiguity issues."""
        logging.info(f"Cleaning DataFrame with columns: {data.columns.tolist() if data is not None else 'None'}")
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
            logging.warning(f"Error cleaning DataFrame: {str(e)}")
            try:
                data = data.reset_index(drop=True)
                for col in ['Datetime', 'Date']:
                    if col in data.columns and data[col].dtype == 'int64':
                        data = data.drop(columns=[col], errors='ignore')
            except:
                pass
        logging.info(f"Cleaned DataFrame columns: {data.columns.tolist() if data is not None else 'None'}")
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
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 5px;
            }
            h3 {
                color: #3b6e9c;
                font-size: 1.4em;
            }
            hr {
                border: 0;
                border-top: 1px solid #e9ecef;
                margin: 20px 0;
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
        
        # Debug historical_data before cleaning
        logging.info(f"Input historical_data columns: {historical_data.columns.tolist() if historical_data is not None else 'None'}")
        logging.info(f"Input historical_data shape: {historical_data.shape if historical_data is not None else 'None'}")
        if historical_data is not None and not historical_data.empty:
            logging.info(f"Input historical_data head:\n{historical_data.head().to_string()}")
        
        historical_data = self._clean_dataframe(historical_data)
        
        # Debug historical_data after cleaning
        logging.info(f"Cleaned historical_data columns: {historical_data.columns.tolist() if historical_data is not None else 'None'}")
        logging.info(f"Cleaned historical_data shape: {historical_data.shape if historical_data is not None else 'None'}")
        
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
        
        <div id="ExecutiveSummary">
            {self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)}
        </div>
        <hr>
        <div id="TechnicalAnalysis">
            {self._generate_technical_charts_section(tech_indicators)}
        </div>
        <hr>
        <div id="TradingSignals">
            {self._generate_trading_signals_section(tech_indicators)}
        </div>
        <hr>
        <div id="PriceCharts">
            {self._generate_price_charts_section(visualizations)}
        </div>
        <hr>
        <div id="PerformanceMetrics">
            {self._generate_performance_metrics(historical_data)}
        </div>
        <hr>
        <div id="RiskAnalysis">
            {self._generate_risk_analysis(analytics, historical_data)}
        </div>
"""
        if predictions is not None and report_type in ["full", "predictions"]:
            html_content += f"""
        <hr>
        <div id="PricePredictions">
            {self._generate_prediction_charts_section(historical_data)}
        </div>
"""
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += f"""
        <hr>
        <div id="3DVisualizations">
            {self._generate_3d_charts_section(visualizations)}
        </div>
"""
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += f"""
        <hr>
        <div id="AdvancedAnalytics">
            {self._generate_advanced_analytics_section(advanced_analytics)}
        </div>
"""
        html_content += f"""
        <footer>
            <p>This report was generated automatically by the Financial Analysis Application</p>
            <p>Data analysis period: {str(historical_data.index[0])[:10] if historical_data is not None and not historical_data.empty else 'N/A'} to {str(historical_data.index[-1])[:10] if historical_data is not None and not historical_data.empty else 'N/A'}</p>
        </footer>
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
        """Generate executive summary section with robust error handling."""
        html_section = """
        <h2>📊 Executive Summary</h2>
"""
        logging.info(f"Generating executive summary for symbol: {symbol}")
        
        try:
            # Validate inputs
            if not isinstance(symbol, str) or not symbol.strip():
                logging.error("Invalid stock symbol provided")
                html_section += """
        <p class="error">Invalid or missing stock symbol.</p>
"""
                return html_section
            
            if data is None or data.empty:
                logging.error("Historical data is None or empty")
                html_section += """
        <p class="error">No historical data provided for executive summary.</p>
"""
                return html_section
            
            required_columns = ['Close', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}. Available: {data.columns.tolist()}")
                html_section += f"""
        <p class="error">Missing required columns: {html.escape(str(missing_columns))}. Available: {html.escape(str(data.columns.tolist()))}</p>
"""
                return html_section
            
            # Validate data types
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logging.error(f"Non-numeric data in column {col}: {data[col].dtype}")
                    html_section += f"""
        <p class="error">Column '{html.escape(col)}' contains non-numeric data (type: {html.escape(str(data[col].dtype))}).</p>
"""
                    return html_section
            
            # Handle missing or NaN values
            if data['Close'].isna().any():
                logging.warning("NaN values found in 'Close' column")
                data = data.dropna(subset=['Close'])
                if data.empty:
                    html_section += """
        <p class="error">No valid 'Close' prices after removing NaNs.</p>
"""
                    return html_section
            
            # Calculate metrics with fallbacks
            current_price = data['Close'].iloc[-1] if not data['Close'].empty else "N/A"
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0
            price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
            high_52week = data['High'].max() if not data['High'].empty else "N/A"
            low_52week = data['Low'].min() if not data['Low'].empty else "N/A"
            avg_volume = data['Volume'].mean() if not data['Volume'].empty else "N/A"
            
            # Format values, handling N/A
            current_price_str = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price)
            high_52week_str = f"${high_52week:.2f}" if isinstance(high_52week, (int, float)) else str(high_52week)
            low_52week_str = f"${low_52week:.2f}" if isinstance(low_52week, (int, float)) else str(low_52week)
            avg_volume_str = f"{avg_volume:,.0f}" if isinstance(avg_volume, (int, float)) else str(avg_volume)
            
            # Add technical indicators if available
            tech_summary = ""
            if tech_indicators is not None and hasattr(tech_indicators, 'get_trading_signals'):
                try:
                    signals = tech_indicators.get_trading_signals()
                    if signals:
                        signal_items = []
                        for indicator, signal_data in signals.items():
                            signal_type = signal_data.get('signal', 'Unknown')
                            signal_class = 'signal-hold'
                            if 'buy' in signal_type.lower():
                                signal_class = 'signal-buy'
                            elif 'sell' in signal_type.lower():
                                signal_class = 'signal-sell'
                            signal_items.append(f"<li>{html.escape(indicator)}: <span class='{signal_class}'>{html.escape(signal_type)}</span></li>")
                        tech_summary = f"""
        <h3>Technical Signals</h3>
        <ul>{''.join(signal_items)}</ul>
"""
                        logging.info("Added technical signals to executive summary")
                except Exception as e:
                    logging.warning(f"Failed to get technical signals: {str(e)}")
            
            # Build table
            html_section += f"""
        <table class="summary-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><th>Current Price</th><td>{current_price_str}</td></tr>
            <tr><th>Daily Change</th><td>{price_change:+.2f} ({price_change_pct:+.2f}%)</td></tr>
            <tr><th>52-Week High</th><td>{high_52week_str}</td></tr>
            <tr><th>52-Week Low</th><td>{low_52week_str}</td></tr>
            <tr><th>Average Volume</th><td>{avg_volume_str}</td></tr>
        </table>
{tech_summary}
"""
            
        except Exception as e:
            logging.error(f"Error in executive summary: {str(e)}")
            html_section += f"""
        <p class="error">Error generating executive summary: {html.escape(str(e))}</p>
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
"""
        logging.info("Generating risk analysis section")
        
        try:
            # Validate input data
            if data is None or data.empty:
                logging.error("Historical data is None or empty")
                html_section += """
        <p class="error">No historical data provided for risk analysis.</p>
        </table>
"""
                return html_section
            
            if 'Close' not in data.columns:
                logging.error(f"'Close' column missing in data. Available columns: {data.columns.tolist()}")
                html_section += f"""
        <p class="error">Required 'Close' column not found in historical data. Available columns: {html.escape(str(data.columns.tolist()))}</p>
        </table>
"""
                return html_section
            
            # Ensure 'Close' is numeric
            if not pd.api.types.is_numeric_dtype(data['Close']):
                logging.error(f"'Close' column is not numeric: {data['Close'].dtype}")
                html_section += f"""
        <p class="error">'Close' column contains non-numeric data (type: {html.escape(str(data['Close'].dtype))}).</p>
        </table>
"""
                return html_section
            
            # Check for sufficient data
            if len(data) < 2:
                logging.error(f"Insufficient data: {len(data)} rows")
                html_section += """
        <p class="error">Insufficient data for risk analysis (requires at least 2 rows).</p>
        </table>
"""
                return html_section
            
            # Calculate returns and handle NaNs
            returns = data['Close'].pct_change().dropna()
            if returns.empty:
                logging.error("No valid returns calculated (possibly due to NaNs or constant prices)")
                html_section += """
        <p class="error">Unable to calculate returns (data may contain NaNs or constant prices).</p>
        </table>
"""
                return html_section
            
            # Calculate metrics
            volatility = returns.std() * (252 ** 0.5) * 100
            if pd.isna(volatility):
                logging.error("Volatility calculation resulted in NaN")
                html_section += """
        <p class="error">Volatility calculation failed (resulted in NaN).</p>
        </table>
"""
                return html_section
            
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
            if pd.isna(max_drawdown):
                logging.error("Max drawdown calculation resulted in NaN")
                html_section += """
        <p class="error">Maximum drawdown calculation failed (resulted in NaN).</p>
        </table>
"""
                return html_section
            
            # Check if analytics provides additional metrics
            analytics_metrics = ""
            if analytics is not None and hasattr(analytics, 'get_risk_metrics'):
                try:
                    extra_metrics = analytics.get_risk_metrics()
                    if extra_metrics:
                        for metric, value in extra_metrics.items():
                            analytics_metrics += f"""
            <tr><th>{html.escape(metric)}</th><td>{html.escape(str(value))}</td></tr>
"""
                        logging.info("Added extra risk metrics from analytics")
                except Exception as e:
                    logging.warning(f"Failed to get risk metrics from analytics: {str(e)}")
            
            # Build table
            html_section += """
        <table class="summary-table">
            <tr><th>Metric</th><th>Value</th></tr>
"""
            html_section += f"""
            <tr><th>Annualized Volatility</th><td>{volatility:.2f}%</td></tr>
            <tr><th>Maximum Drawdown</th><td>{max_drawdown:.2f}%</td></tr>
{analytics_metrics}
        </table>
        <h3>Risk Assessment</h3>
"""
            # Risk assessment
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
            logging.error(f"Error in risk analysis: {str(e)}")
            html_section += f"""
        <table class="summary-table"></table>
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
            <summary>Industry Analysis</summary>
            <table class="summary-table">{industry_data.head(10).to_html(classes="industry-table", escape=False)}</table>
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
