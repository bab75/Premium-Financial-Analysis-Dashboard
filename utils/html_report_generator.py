"""
HTML Report Generator for Financial Analysis
Creates comprehensive downloadable HTML reports with interactive charts
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import StringIO

class HTMLReportGenerator:
    """Generate comprehensive HTML reports with interactive charts and analysis."""
    
    def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to resolve index/column ambiguity issues."""
        if data is None:
            return data
        
        # Create a copy to avoid modifying original data
        data = data.copy()
        
        try:
            # Force remove 'Datetime' column if it exists as both index and column
            if 'Datetime' in data.columns:
                # Check if index appears to be datetime-like
                if isinstance(data.index, pd.DatetimeIndex) or (
                    hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype).lower()
                ):
                    # Remove the duplicate column
                    data = data.drop(columns=['Datetime'], errors='ignore')
                else:
                    # Set as index if index is not datetime
                    try:
                        data = data.set_index('Datetime')
                    except:
                        # If setting index fails, just remove the column
                        data = data.drop(columns=['Datetime'], errors='ignore')
            
            # Similar handling for 'Date' column
            if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                try:
                    # Try to use Date column as index
                    if data['Date'].dtype == 'int64':
                        # Skip integer date columns that are just counters
                        pass
                    else:
                        data = data.set_index('Date')
                        data.index = pd.to_datetime(data.index)
                except:
                    pass  # Keep original if conversion fails
            
            # Ensure we have some kind of reasonable index
            if not isinstance(data.index, pd.DatetimeIndex):
                # Reset to default integer index if no datetime index exists
                data = data.reset_index(drop=True)
                
        except Exception as e:
            # If all cleaning fails, return a simplified version
            try:
                data = data.reset_index(drop=True)
                # Remove any problematic columns that might cause ambiguity
                for col in ['Datetime', 'Date']:
                    if col in data.columns and data[col].dtype == 'int64':
                        data = data.drop(columns=[col], errors='ignore')
            except:
                pass  # Return original data if all fixes fail
        
        return data
    
    def __init__(self):
        self.css_styles = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            background: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #4a5568;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #718096;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .signal-card {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }
        .signal-buy {
            background: #f0fff4;
            border-color: #48bb78;
        }
        .signal-sell {
            background: #fff5f5;
            border-color: #f56565;
        }
        .signal-hold {
            background: #fffaf0;
            border-color: #ed8936;
        }
        .chart-container {
            margin: 20px 0;
            padding: 10px;
            background: #fafafa;
            border-radius: 8px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .summary-table th,
        .summary-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .summary-table th {
            background-color: #edf2f7;
            font-weight: 600;
            color: #4a5568;
        }
        .summary-table tr:hover {
            background-color: #f7fafc;
        }
        </style>
        """
    
    def generate_comprehensive_report(self, 
                                    stock_symbol: str,
                                    historical_data: pd.DataFrame,
                                    tech_indicators,
                                    analytics,
                                    visualizations,
                                    prediction_data=None,
                                    prediction_days=None,
                                    advanced_analytics=None,
                                    report_type: str = "full") -> str:
        """Generate a comprehensive HTML report with all analysis components."""
        
        # Clean data to resolve ambiguity issues
        historical_data = self._clean_dataframe(historical_data)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Financial Analysis Report - {stock_symbol}</title>
            {self.css_styles}
        </head>
        <body>
        """
        
        # Header
        html_content += f"""
        <div class="header">
            <h1>Financial Analysis Report</h1>
            <p>Stock Symbol: <strong>{stock_symbol}</strong> | Generated: {timestamp}</p>
        </div>
        """
        
        # Executive Summary
        html_content += self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)
        
        # Technical Indicators Charts
        html_content += self._generate_technical_charts_section(tech_indicators)
        
        # Trading Signals
        html_content += self._generate_trading_signals_section(tech_indicators)
        
        # Price Visualization Charts
        html_content += self._generate_price_charts_section(visualizations)
        
        # Performance Metrics
        html_content += self._generate_performance_metrics(historical_data)
        
        # Risk Analysis
        html_content += self._generate_risk_analysis(analytics, historical_data)
        
        # Add prediction charts if available
        if prediction_data is not None and report_type in ["full", "predictions"]:
            html_content += self._generate_prediction_charts_section(prediction_data, prediction_days, historical_data)
        
        # Add 3D visualization charts if available
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_3d_charts_section(visualizations)
        
        # Add advanced analytics if available
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_advanced_analytics_section(advanced_analytics)
        
        # Footer
        html_content += f"""
        <div class="footer">
            <p>This report was generated automatically by the Financial Analysis Application</p>
            <p>Data analysis period: {str(historical_data.index[0])[:10]} to {str(historical_data.index[-1])[:10]}</p>
        </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
        """Generate executive summary section."""
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
        
        # Calculate key metrics
        high_52week = data['High'].max()
        low_52week = data['Low'].min()
        avg_volume = data['Volume'].mean()
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${current_price:.2f}</div>
                    <div class="metric-label">Current Price</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#48bb78' if price_change >= 0 else '#f56565'}">
                        {price_change:+.2f} ({price_change_pct:+.2f}%)
                    </div>
                    <div class="metric-label">Daily Change</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${high_52week:.2f}</div>
                    <div class="metric-label">52-Week High</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${low_52week:.2f}</div>
                    <div class="metric-label">52-Week Low</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_volume:,.0f}</div>
                    <div class="metric-label">Avg Volume</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_technical_charts_section(self, tech_indicators) -> str:
        """Generate technical analysis charts section."""
        html_section = """
        <div class="section">
            <h2>📈 Technical Analysis Charts</h2>
            <p>Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.</p>
        """
        
        try:
            # Moving Averages Chart
            ma_chart = tech_indicators.create_moving_averages_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Moving Averages</h3>
                {ma_chart.to_html(include_plotlyjs='inline', div_id='ma-chart')}
            </div>
            """
            
            # RSI Chart
            rsi_chart = tech_indicators.create_rsi_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Relative Strength Index (RSI)</h3>
                {rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')}
            </div>
            """
            
            # MACD Chart
            macd_chart = tech_indicators.create_macd_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>MACD Analysis</h3>
                {macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')}
            </div>
            """
            
            # Bollinger Bands Chart
            bb_chart = tech_indicators.create_bollinger_bands_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Bollinger Bands</h3>
                {bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')}
            </div>
            """
            
        except Exception as e:
            html_section += f"<p>Error generating technical charts: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def _generate_trading_signals_section(self, tech_indicators) -> str:
        """Generate trading signals analysis section."""
        html_section = """
        <div class="section">
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
                <div class="signal-card {signal_class}">
                    <h4>{indicator}</h4>
                    <p><strong>Signal:</strong> {signal_data.get('signal', 'Unknown')}</p>
                    <p><strong>Strength:</strong> {strength}</p>
                </div>
                """
                
        except Exception as e:
            html_section += f"<p>Error generating trading signals: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def _generate_price_charts_section(self, visualizations) -> str:
        """Generate price visualization charts section."""
        html_section = """
        <div class="section">
            <h2>💹 Price Analysis Charts</h2>
        """
        
        try:
            # Candlestick Chart
            candlestick_chart = visualizations.create_candlestick_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Candlestick Chart</h3>
                {candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')}
            </div>
            """
            
            # Price Trends Chart
            trends_chart = visualizations.create_price_trends_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Price Trends</h3>
                {trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')}
            </div>
            """
            
            # Volume Analysis Chart
            volume_chart = visualizations.create_volume_chart()
            html_section += f"""
            <div class="chart-container">
                <h3>Volume Analysis</h3>
                {volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')}
            </div>
            """
            
        except Exception as e:
            html_section += f"<p>Error generating price charts: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
        """Generate performance metrics table."""
        html_section = """
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        try:
            # Calculate metrics
            total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized
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
                    <td><strong>{metric}</strong></td>
                    <td>{value}</td>
                    <td>{description}</td>
                </tr>
                """
                
        except Exception as e:
            html_section += f"<tr><td colspan='3'>Error calculating metrics: {str(e)}</td></tr>"
        
        html_section += """
                </tbody>
            </table>
        </div>
        """
        return html_section
    
    def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        html_section = """
        <div class="section">
            <h2>⚠️ Risk Analysis</h2>
        """
        
        try:
            # Calculate risk metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
            
            html_section += f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{volatility:.2f}%</div>
                    <div class="metric-label">Annualized Volatility</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #f56565">{max_drawdown:.2f}%</div>
                    <div class="metric-label">Maximum Drawdown</div>
                </div>
            </div>
            
            <h3>Risk Assessment</h3>
            <p>
            """
            
            if volatility > 30:
                html_section += "🔴 <strong>High Risk:</strong> This stock shows high volatility. Suitable for experienced traders with high risk tolerance."
            elif volatility > 20:
                html_section += "🟡 <strong>Medium Risk:</strong> Moderate volatility. Suitable for balanced investment strategies."
            else:
                html_section += "🟢 <strong>Low Risk:</strong> Relatively stable price movements. Suitable for conservative investors."
            
            html_section += "</p>"
            
        except Exception as e:
            html_section += f"<p>Error calculating risk metrics: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_report_{timestamp}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_prediction_charts_section(self, prediction_data, prediction_days, historical_data) -> str:
        """Generate prediction charts section for HTML report using stored prediction data."""
        html_section = """
        <div class="section">
            <h2>📈 Price Predictions</h2>
        """
        
        if prediction_data is None:
            html_section += """
            <p>No price predictions generated. Generate predictions in the application first to include them in the report.</p>
            </div>
            """
            return html_section
        
        html_section += f"<p>Price predictions for {prediction_days} days using {prediction_data.get('method', 'selected')} method.</p>"
        
        try:
            # Get stored prediction results
            pred_prices = prediction_data.get('prices', [])
            method = prediction_data.get('method', 'technical_analysis')
            confidence_data = prediction_data.get('confidence', {})
            disclaimer = prediction_data.get('disclaimer', '')
            
            if pred_prices and len(pred_prices) > 0:
                # Create prediction chart using stored data
                import plotly.graph_objects as go
                import pandas as pd
                from datetime import datetime, timedelta
                
                # Get recent historical prices for context
                recent_data = historical_data.tail(20)
                historical_dates = recent_data.index
                historical_prices = recent_data['Close'].values
                
                # Generate future dates for predictions
                if isinstance(historical_dates, pd.DatetimeIndex) and len(historical_dates) > 0:
                    last_date = historical_dates[-1]
                else:
                    last_date = pd.Timestamp.now()
                
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(pred_prices), freq='D')
                
                # Create chart
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Predicted prices
                method_name = {
                    'technical_analysis': 'Technical Analysis',
                    'linear_trend': 'Linear Trend',
                    'moving_average': 'Moving Average'
                }.get(method, method.replace('_', ' ').title())
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name=f'{method_name} Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>Predicted: $%{y:.2f}<extra></extra>'
                ))
                
                # Update layout with MM-DD-YYYY format
                fig.update_layout(
                    title=f'{method_name} Price Prediction - {prediction_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified',
                    showlegend=True,
                    height=400,
                    xaxis=dict(
                        type='date',
                        showticklabels=True,
                        showgrid=True,
                        hoverformat='%m-%d-%Y'
                    )
                )
                
                chart_html = fig.to_html(include_plotlyjs=False, div_id=f"prediction_chart")
                
                html_section += f"""
                <div class="chart-container">
                    <h3>{method_name} Prediction Chart</h3>
                    {chart_html}
                </div>
                """
                
                # Add prediction table for easy readability
                current_date = datetime.now()
                pred_dates = [current_date + timedelta(days=i+1) for i in range(len(pred_prices))]
                
                html_section += f"""
                <div class="chart-container">
                    <h3>Predicted Prices Table</h3>
                    <div class="summary-table">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <th style="padding: 8px; border: 1px solid #ddd;">Date</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Day</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Predicted Price</th>
                            </tr>
                """
                
                for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
                    html_section += f"""
                            <tr>
                                <td style="padding: 8px; border: 1px solid #ddd;">{date.strftime('%Y-%m-%d')}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">Day {i+1}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">${price:.2f}</td>
                            </tr>
                    """
                
                html_section += """
                        </table>
                    </div>
                </div>
                """
                
                # Add prediction metrics and disclaimer
                if confidence_data:
                    # Fix confidence level formatting - handle string values
                    confidence_level = confidence_data.get('confidence_level', 'N/A')
                    if isinstance(confidence_level, str) and '%' in confidence_level:
                        confidence_display = confidence_level
                    else:
                        try:
                            confidence_display = f"{float(confidence_level):.1f}%"
                        except:
                            confidence_display = str(confidence_level)
                    
                    volatility_risk = confidence_data.get('volatility_risk', 0)
                    try:
                        volatility_display = f"{float(volatility_risk):.2f}%"
                    except:
                        volatility_display = str(volatility_risk)
                    
                    html_section += f"""
                    <div class="chart-container">
                        <h3>Prediction Metrics & Reliability</h3>
                        <div class="summary-table">
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr><th style="padding: 8px; border: 1px solid #ddd;">Metric</th><th style="padding: 8px; border: 1px solid #ddd;">Value</th></tr>
                                <tr><td style="padding: 8px; border: 1px solid #ddd;">Confidence Level</td><td style="padding: 8px; border: 1px solid #ddd;">{confidence_display}</td></tr>
                                <tr><td style="padding: 8px; border: 1px solid #ddd;">Trend Strength</td><td style="padding: 8px; border: 1px solid #ddd;">{confidence_data.get('trend_strength', 'N/A')}</td></tr>
                                <tr><td style="padding: 8px; border: 1px solid #ddd;">Data Quality</td><td style="padding: 8px; border: 1px solid #ddd;">{confidence_data.get('data_quality', 'N/A')}</td></tr>
                                <tr><td style="padding: 8px; border: 1px solid #ddd;">Volatility Risk</td><td style="padding: 8px; border: 1px solid #ddd;">{volatility_display}</td></tr>
                            </table>
                        </div>
                        <div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
                            <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">{disclaimer}</pre>
                        </div>
                    </div>
                    """
            else:
                html_section += "<p>No prediction data available.</p>"
                
        except Exception as e:
            html_section += f"<p>Error generating prediction charts: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def _generate_3d_charts_section(self, visualizations) -> str:
        """Generate 3D visualization charts section for HTML report."""
        html_section = """
        <div class="section">
            <h2>📊 3D Visualizations</h2>
            <p>Interactive three-dimensional analysis for comprehensive market insights.</p>
        """
        
        try:
            # Check if visualizations has 3D chart methods
            if hasattr(visualizations, 'get_3d_price_volume_chart'):
                chart = visualizations.get_3d_price_volume_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_price_volume")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>3D Price-Volume Analysis</h3>
                        <p>Three-dimensional visualization of price movements, volume, and time relationships.</p>
                        {chart_html}
                    </div>
                    """
            
            if hasattr(visualizations, 'get_3d_technical_surface'):
                chart = visualizations.get_3d_technical_surface()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_technical_surface")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>3D Technical Indicator Surface</h3>
                        <p>Surface plot showing relationships between multiple technical indicators.</p>
                        {chart_html}
                    </div>
                    """
            
            if hasattr(visualizations, 'get_3d_market_dynamics'):
                chart = visualizations.get_3d_market_dynamics()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_market_dynamics")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>3D Market Dynamics</h3>
                        <p>Multi-dimensional view of market behavior and trading patterns.</p>
                        {chart_html}
                    </div>
                    """
            
        except Exception as e:
            html_section += f"<p>Error generating 3D charts: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
    
    def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
        """Generate advanced analytics section for HTML report."""
        html_section = """
        <div class="section">
            <h2>🎯 Advanced Analytics</h2>
            <p>Comprehensive sector analysis, correlations, and market intelligence.</p>
        """
        
        try:
            # Sector Performance Analysis
            if hasattr(advanced_analytics, 'create_sector_performance_chart'):
                chart = advanced_analytics.create_sector_performance_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="sector_performance")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>Sector Performance Analysis</h3>
                        <p>Comparative performance across different market sectors.</p>
                        {chart_html}
                    </div>
                    """
            
            # Correlation Heatmap
            if hasattr(advanced_analytics, 'create_correlation_heatmap'):
                chart = advanced_analytics.create_correlation_heatmap()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="correlation_heatmap")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>Market Correlation Analysis</h3>
                        <p>Heat map showing correlations between different market metrics and indicators.</p>
                        {chart_html}
                    </div>
                    """
            
            # Performance Dashboard
            if hasattr(advanced_analytics, 'create_performance_dashboard'):
                chart = advanced_analytics.create_performance_dashboard()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="performance_dashboard")
                    html_section += f"""
                    <div class="chart-container">
                        <h3>Comprehensive Performance Dashboard</h3>
                        <p>Multi-metric dashboard showing key performance indicators and trends.</p>
                        {chart_html}
                    </div>
                    """
            
            # Industry Analysis
            if hasattr(advanced_analytics, 'get_industry_analysis'):
                industry_data = advanced_analytics.get_industry_analysis()
                if industry_data is not None and not industry_data.empty:
                    html_section += """
                    <div class="chart-container">
                        <h3>Industry Analysis Summary</h3>
                        <div class="summary-table">
                    """
                    html_section += industry_data.head(10).to_html(classes="summary-table", escape=False)
                    html_section += "</div></div>"
            
        except Exception as e:
            html_section += f"<p>Error generating advanced analytics: {str(e)}</p>"
        
        html_section += "</div>"
        return html_section
