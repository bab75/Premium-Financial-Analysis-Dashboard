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
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; text-align: center; }
                h2, h3 { color: #444; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .signal-buy { color: green; }
                .signal-sell { color: red; }
                .signal-hold { color: orange; }
                .summary-table { width: 100%; }
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
        
        # Clean data to resolve ambiguity issues
        historical_data = self._clean_dataframe(historical_data)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building HTML
        html_content = "<html><head><title>Financial Analysis Report - " + stock_symbol + "</title>" + self.css_styles + "</head><body>"
        
        # Header
        html_content += "<h1>Financial Analysis Report</h1><p><strong>Stock Symbol:</strong> " + stock_symbol + " | <strong>Generated:</strong> " + timestamp + "</p>"
        
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
        if predictions is not None and report_type in ["full", "predictions"]:
            html_content += self._generate_prediction_charts_section(historical_data)
        
        # Add 3D visualization charts if available
        if visualizations is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_3d_charts_section(visualizations)
        
        # Add advanced analytics if available
        if advanced_analytics is not None and report_type in ["full", "advanced"]:
            html_content += self._generate_advanced_analytics_section(advanced_analytics)
        
        # Footer
        html_content += "<footer><p>This report was generated automatically by the Financial Analysis Application</p><p><strong>Data analysis period:</strong> " + str(historical_data.index[0])[:10] + " to " + str(historical_data.index[-1])[:10] + "</p></footer></body></html>"
        
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
        
        html_section = "<h2>📊 Executive Summary</h2><table>"
        html_section += "<tr><td>Current Price</td><td>$" + str(round(current_price, 2)) + "</td></tr>"
        html_section += "<tr><td>Daily Change</td><td>" + f"{price_change:+.2f} ({price_change_pct:+.2f}%)" + "</td></tr>"
        html_section += "<tr><td>52-Week High</td><td>$" + str(round(high_52week, 2)) + "</td></tr>"
        html_section += "<tr><td>52-Week Low</td><td>$" + str(round(low_52week, 2)) + "</td></tr>"
        html_section += "<tr><td>Average Volume</td><td>" + f"{avg_volume:,.0f}" + "</td></tr>"
        html_section += "</table>"
        
        return html_section
    
    def _generate_technical_charts_section(self, tech_indicators) -> str:
        """Generate technical analysis charts section."""
        html_section = "<h2>📈 Technical Analysis Charts</h2><p>Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.</p>"
        
        try:
            # Moving Averages Chart
            ma_chart = tech_indicators.create_moving_averages_chart()
            html_section += "<h3>Moving Averages</h3>" + ma_chart.to_html(include_plotlyjs='inline', div_id='ma-chart')
            
            # RSI Chart
            rsi_chart = tech_indicators.create_rsi_chart()
            html_section += "<h3>Relative Strength Index (RSI)</h3>" + rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')
            
            # MACD Chart
            macd_chart = tech_indicators.create_macd_chart()
            html_section += "<h3>MACD Analysis</h3>" + macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')
            
            # Bollinger Bands Chart
            bb_chart = tech_indicators.create_bollinger_bands_chart()
            html_section += "<h3>Bollinger Bands</h3>" + bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')
            
        except Exception as e:
            html_section += "<p>Error generating technical charts: " + str(e) + "</p>"
        
        return html_section
    
    def _generate_trading_signals_section(self, tech_indicators) -> str:
        """Generate trading signals analysis section."""
        html_section = "<h2>🎯 Trading Signals & Recommendations</h2>"
        
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
                
                html_section += "<div><h3>" + indicator + "</h3>"
                html_section += "<p>Signal: <span class='" + signal_class + "'>" + signal_data.get('signal', 'Unknown') + "</span></p>"
                html_section += "<p>Strength: " + strength + "</p></div>"
                
        except Exception as e:
            html_section += "<p>Error generating trading signals: " + str(e) + "</p>"
        
        return html_section
    
    def _generate_price_charts_section(self, visualizations) -> str:
        """Generate price visualization charts section."""
        html_section = "<h2>💹 Price Analysis Charts</h2>"
        
        try:
            # Candlestick Chart
            candlestick_chart = visualizations.create_candlestick_chart()
            html_section += "<h3>Candlestick Chart</h3>" + candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')
            
            # Price Trends Chart
            trends_chart = visualizations.create_price_trends_chart()
            html_section += "<h3>Price Trends</h3>" + trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')
            
            # Volume Analysis Chart
            volume_chart = visualizations.create_volume_chart()
            html_section += "<h3>Volume Analysis</h3>" + volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')
            
        except Exception as e:
            html_section += "<p>Error generating price charts: " + str(e) + "</p>"
        
        return html_section
    
    def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
        """Generate performance metrics table."""
        html_section = "<h2>📊 Performance Metrics</h2><table><tr><th>Metric</th><th>Value</th><th>Description</th></tr>"
        
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
                html_section += "<tr><td>" + metric + "</td><td>" + value + "</td><td>" + description + "</td></tr>"
                
        except Exception as e:
            html_section += "<tr><td colspan='3'>Error calculating metrics: " + str(e) + "</td></tr>"
        
        html_section += "</table>"
        return html_section
    
    def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
        """Generate risk analysis section."""
        html_section = "<h2>⚠️ Risk Analysis</h2>"
        
        try:
            # Calculate risk metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
            
            html_section += "<table><tr><td>Annualized Volatility</td><td>" + f"{volatility:.2f}%" + "</td></tr>"
            html_section += "<tr><td>Maximum Drawdown</td><td>" + f"{max_drawdown:.2f}%" + "</td></tr></table>"
            html_section += "<h3>Risk Assessment</h3>"
            
            if volatility > 30:
                html_section += "<p>🔴 High Risk: This stock shows high volatility. Suitable for experienced traders with high risk tolerance.</p>"
            elif volatility > 20:
                html_section += "<p>🟡 Medium Risk: Moderate volatility. Suitable for balanced investment strategies.</p>"
            else:
                html_section += "<p>🟢 Low Risk: Relatively stable price movements. Suitable for conservative investors.</p>"
            
        except Exception as e:
            html_section += "<p>Error calculating risk metrics: " + str(e) + "</p>"
        
        return html_section
    
    def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "financial_analysis_report_" + timestamp + ".html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_prediction_charts_section(self, historical_data) -> str:
        """Generate prediction charts section for HTML report using actual PricePredictions class."""
        html_section = "<h2>📈 Price Predictions</h2><p>Technical analysis, linear trend, and moving average predictions for future price movements.</p>"
        
        try:
            from utils.predictions import PricePredictions
            
            # Initialize predictions with historical data
            predictions = PricePredictions(historical_data)
            
            # Generate predictions for 7 days using all three methods
            prediction_days = 7
            methods = [
                ("technical_analysis", "Technical Analysis Prediction", "Advanced technical indicators and momentum analysis"),
                ("linear_trend", "Linear Trend Prediction", "Statistical trend analysis and regression modeling"),
                ("moving_average", "Moving Average Prediction", "Simple and exponential moving average forecasting")
            ]
            
            for method_key, method_name, method_desc in methods:
                try:
                    pred_data = predictions.predict_prices(prediction_days, method_key)
                    
                    if pred_data and len(pred_data) > 0:
                        # Create prediction chart
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        # Get recent historical prices for context
                        recent_data = historical_data.tail(20)
                        historical_dates = recent_data.index if hasattr(recent_data, 'index') else range(len(recent_data))
                        historical_prices = recent_data['Close'].values
                        
                        # Generate future dates
                        import pandas as pd
                        if isinstance(historical_dates, pd.DatetimeIndex) and len(historical_dates) > 0:
                            last_date = historical_dates[-1]
                        else:
                            last_date = pd.Timestamp.now()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
                        
                        # Create chart
                        fig = go.Figure()
                        
                        # Historical prices
                        fig.add_trace(go.Scatter(
                            x=historical_dates,
                            y=historical_prices,
                            mode='lines',
                            name='Historical Prices',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Predicted prices
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=pred_data,
                            mode='lines+markers',
                            name=method_name,
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=method_name + " - " + str(prediction_days) + " Day Forecast",
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified',
                            showlegend=True,
                            height=400
                        )
                        
                        chart_html = fig.to_html(include_plotlyjs=False, div_id="prediction_" + method_key)
                        
                        html_section += "<h3>" + method_name + "</h3>"
                        html_section += "<p>" + method_desc + "</p>"
                        html_section += chart_html
                        
                except Exception as method_error:
                    html_section += "<p>Error generating " + method_name + ": " + str(method_error) + "</p>"
            
            # Add prediction metrics and disclaimer
            try:
                confidence_data = predictions.calculate_prediction_confidence()
                disclaimer = predictions.get_prediction_disclaimer()
                
                # Format metrics with type checking
                metrics = [
                    ("Confidence Level", confidence_data.get('confidence_level', 'N/A'), "%"),
                    ("Trend Strength", confidence_data.get('trend_strength', 'N/A'), ""),
                    ("Data Quality", confidence_data.get('data_quality', 'N/A'), ""),
                    ("Volatility Risk", confidence_data.get('volatility_risk', 0), "%")
                ]
                
                html_section += "<h3>Prediction Metrics & Reliability</h3><table><tr><th>Metric</th><th>Value</th></tr>"
                
                for metric, value, suffix in metrics:
                    formatted_value = f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else str(value)
                    html_section += "<tr><td>" + metric + "</td><td>" + formatted_value + "</td></tr>"
                
                html_section += "</table>"
                html_section += "<p><em>" + disclaimer + "</em></p>"
                
            except Exception as metrics_error:
                html_section += "<p>Error generating prediction metrics: " + str(metrics_error) + "</p>"
            
        except Exception as e:
            html_section += "<p>Error initializing predictions: " + str(e) + "</p>"
        
        return html_section
    
    def _generate_3d_charts_section(self, visualizations) -> str:
        """Generate 3D visualization charts section for HTML report."""
        html_section = "<h2>📊 3D Visualizations</h2><p>Interactive three-dimensional analysis for comprehensive market insights.</p>"
        
        try:
            # Check if visualizations has 3D chart methods
            if hasattr(visualizations, 'get_3d_price_volume_chart'):
                chart = visualizations.get_3d_price_volume_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_price_volume")
                    html_section += "<h3>3D Price-Volume Analysis</h3><p>Three-dimensional visualization of price movements, volume, and time relationships.</p>" + chart_html
            
            if hasattr(visualizations, 'get_3d_technical_surface'):
                chart = visualizations.get_3d_technical_surface()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_technical_surface")
                    html_section += "<h3>3D Technical Indicator Surface</h3><p>Surface plot showing relationships between multiple technical indicators.</p>" + chart_html
            
            if hasattr(visualizations, 'get_3d_market_dynamics'):
                chart = visualizations.get_3d_market_dynamics()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_market_dynamics")
                    html_section += "<h3>3D Market Dynamics</h3><p>Multi-dimensional view of market behavior and trading patterns.</p>" + chart_html
            
        except Exception as e:
            html_section += "<p>Error generating 3D charts: " + str(e) + "</p>"
        
        return html_section
    
    def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
        """Generate advanced analytics section for HTML report."""
        html_section = "<h2>🎯 Advanced Analytics</h2><p>Comprehensive sector analysis, correlations, and market intelligence.</p>"
        
        try:
            # Sector Performance Analysis
            if hasattr(advanced_analytics, 'create_sector_performance_chart'):
                chart = advanced_analytics.create_sector_performance_chart()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="sector_performance")
                    html_section += "<h3>Sector Performance Analysis</h3><p>Comparative performance across different market sectors.</p>" + chart_html
            
            # Correlation Heatmap
            if hasattr(advanced_analytics, 'create_correlation_heatmap'):
                chart = advanced_analytics.create_correlation_heatmap()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="correlation_heatmap")
                    html_section += "<h3>Market Correlation Analysis</h3><p>Heat map showing correlations between different market metrics and indicators.</p>" + chart_html
            
            # Performance Dashboard
            if hasattr(advanced_analytics, 'create_performance_dashboard'):
                chart = advanced_analytics.create_performance_dashboard()
                if chart:
                    chart_html = chart.to_html(include_plotlyjs=False, div_id="performance_dashboard")
                    html_section += "<h3>Comprehensive Performance Dashboard</h3><p>Multi-metric dashboard showing key performance indicators and trends.</p>" + chart_html
            
            # Industry Analysis
            if hasattr(advanced_analytics, 'get_industry_analysis'):
                industry_data = advanced_analytics.get_industry_analysis()
                if industry_data is not None and not industry_data.empty:
                    html_section += "<h3>Industry Analysis Summary</h3><table class='summary-table'>"
                    html_section += industry_data.head(10).to_html(classes="summary-table", escape=False)
                    html_section += "</table>"
            
        except Exception as e:
            html_section += "<p>Error generating advanced analytics: " + str(e) + "</p>"
        
        return html_section

