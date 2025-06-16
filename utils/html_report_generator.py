import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
import pytz
from typing import Dict, List, Optional
import json
import html
import logging
import numpy as np
import uuid
import os

logging.basicConfig(level=logging.INFO)

class HTMLReportGenerator:
    """Generate compact HTML reports with React and external chart data."""
    
    def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to resolve index/column ambiguity issues."""
        if data is None or data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        try:
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
                data = data.dropna(subset=['Datetime'])
                data = data.set_index('Datetime')
            elif data.index.name == 'Datetime' or 'Date' in data.columns:
                if 'Date' in data.columns:
                    data['Datetime'] = pd.to_datetime(data['Date'], errors='coerce')
                    data = data.drop(columns=['Date'])
                data['Datetime'] = pd.to_datetime(data.index, errors='coerce')
                if data['Datetime'].isna().all():
                    data['Datetime'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                data = data.set_index('Datetime')
            else:
                data['Datetime'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                data = data.set_index('Datetime')
            
            if len(data) > 252 * 2:  # Limit to 2 years of data
                data = data.tail(252 * 2)
            
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close'] if 'Close' in data.columns else 0
            for col in ['Dividends', 'Stock Splits']:
                if col not in data.columns:
                    data[col] = 0
            data = data.sort_index()
            return data
        except Exception as e:
            logging.error(f"Data cleaning failed: {str(e)}")
            return data.reset_index(drop=True)

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.js_script = """
        <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            'primary': '#1e3a8a',
                            'secondary': '#3b82f6',
                            'buy': '#10b981',
                            'sell': '#ef4444',
                            'neutral': '#64748b',
                        },
                    },
                },
            };
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
                                    additional_figures=None,
                                    report_type: str = "full") -> str:
        """Generate a React-based HTML report with external chart data."""
        logging.info(f"Generating report for {stock_symbol}, report_type: {report_type}")
        
        historical_data = self._clean_dataframe(historical_data)
        local_tz = pytz.timezone('America/New_York')
        timestamp = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        stock_symbol = html.escape(str(stock_symbol))
        timestamp = html.escape(str(timestamp))
        
        if historical_data.empty:
            return self._generate_empty_report(stock_symbol, timestamp)

        # Save chart data to JSON files
        chart_files = self._save_chart_data(additional_figures, historical_data, predictions, report_type)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Analysis Report - {stock_symbol}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.js_script}
        </head>
        <body class="bg-gradient-to-br from-blue-100 to-purple-100 font-sans text-gray-800">
            <div id="root"></div>
            <script type="text/babel">
                const {{ useState, useEffect }} = React;

                function MetricCard({{ title, value }}) {{
                    return (
                        <div className="bg-white p-4 rounded-lg shadow-sm hover:scale-105 transition-transform">
                            <h4 className="font-semibold text-primary">{{title}}</h4>
                            <p>{{value}}</p>
                        </div>
                    );
                }}

                function ChartContainer({{ title, chartId, chartUrl }}) {{
                    const [isOpen, setIsOpen] = useState(false);
                    useEffect(() => {{
                        if (isOpen && chartUrl) {{
                            fetch(chartUrl)
                                .then(response => response.json())
                                .then(data => {{
                                    Plotly.newPlot(chartId, data.data, data.layout, {{ displayModeBar: false }});
                                }})
                                .catch(error => console.error('Error loading chart:', error));
                        }}
                    }}, [isOpen, chartUrl]);
                    return (
                        <details className="mb-4 bg-white rounded-lg shadow-sm">
                            <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg" onClick={() => setIsOpen(!isOpen)}>
                                {{title}}
                            </summary>
                            <div className="p-4">
                                <div id={{chartId}} className="h-96"></div>
                            </div>
                        </details>
                    );
                }}

                function Report() {{
                    return (
                        <div className="max-w-7xl mx-auto p-6">
                            <h1 className="text-4xl font-bold text-center bg-gradient-to-r from-primary to-secondary text-transparent bg-clip-text mb-4">
                                Financial Analysis Report: {stock_symbol}
                            </h1>
                            <p className="text-center text-gray-600 mb-8">Generated on: {timestamp}</p>
                            
                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Overview</h2>
                                {self._generate_overview(historical_data)}
                            </div>

                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Advanced Visualizations</h2>
                                {self._generate_visualizations(chart_files.get('additional_figures', {}))}
                            </div>

                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Technical Indicators</h2>
                                {self._generate_technical_indicators(tech_indicators)}
                            </div>

                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Trading Insights</h2>
                                {self._generate_trading_insights(tech_indicators, analytics)}
                            </div>

                            {predictions is not None and len(historical_data) > 50 and report_type in ["full", "predictions"] and f"""
                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Price Predictions</h2>
                                {self._generate_price_predictions(historical_data, predictions, chart_files.get('predictions', {}))}
                            </div>
                            """}

                            {advanced_analytics is not None and report_type in ["full", "advanced"] and f"""
                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Comparative Analysis</h2>
                                {self._generate_comparative_analysis(advanced_analytics)}
                            </div>
                            """}

                            {not historical_data.empty and f"""
                            <div className="mb-8 bg-white p-6 rounded-lg shadow-sm">
                                <h2 className="text-2xl font-semibold text-primary mb-4">Data Summary</h2>
                                {self._generate_data_summary(historical_data)}
                            </div>
                            """}

                            <div className="text-sm italic text-neutral">
                                <p><strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute financial advice.</p>
                                <p>Data analysis period: {historical_data.index[0].strftime('%Y-%m-%d') if not historical_data.empty else 'N/A'} to {historical_data.index[-1].strftime('%Y-%m-%d') if not historical_data.empty else 'N/A'}</p>
                            </div>
                        </div>
                    );
                }}

                ReactDOM.render(<Report />, document.getElementById('root'));
            </script>
            <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/Babel.min.js"></script>
        </body>
        </html>
        """
        return self.save_report_to_file(html_content.encode('utf-8'))

    def _generate_empty_report(self, stock_symbol: str, timestamp: str) -> str:
        """Generate a minimal report for empty data."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Analysis Report - {stock_symbol}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.js_script}
        </head>
        <body class="bg-gradient-to-br from-blue-100 to-purple-100 font-sans text-gray-800">
            <div id="root"></div>
            <script type="text/babel">
                function Report() {{
                    return (
                        <div className="max-w-7xl mx-auto p-6">
                            <h1 className="text-4xl font-bold text-center bg-gradient-to-r from-primary to-secondary text-transparent bg-clip-text mb-4">
                                Financial Analysis Report: {stock_symbol}
                            </h1>
                            <p className="text-center text-gray-600 mb-8">Generated on: {timestamp}</p>
                            <div className="bg-white p-6 rounded-lg shadow-sm">
                                <p>No historical data available to generate the report.</p>
                            </div>
                        </div>
                    );
                }}
                ReactDOM.render(<Report />, document.getElementById('root'));
            </script>
            <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/Babel.min.js"></script>
        </body>
        </html>
        """
        return self.save_report_to_file(html_content.encode('utf-8'))

    def _save_chart_data(self, additional_figures: Dict, historical_data: pd.DataFrame, predictions, report_type: str) -> Dict:
        """Save chart data to external JSON files."""
        chart_files = {'additional_figures': {}, 'predictions': {}}
        if additional_figures:
            for key, title, chart_id in [
                ('candlestick', 'Candlestick Chart', f'candlestick-{uuid.uuid4()}'),
                ('price_trends', 'Price Trends', f'price-trends-{uuid.uuid4()}'),
                ('volume_analysis', 'Volume Analysis', f'volume-{uuid.uuid4()}'),
                ('moving_averages', 'Moving Averages', f'ma-{uuid.uuid4()}'),
                ('rsi_chart', 'RSI Chart', f'rsi-{uuid.uuid4()}'),
                ('macd_chart', 'MACD Chart', f'macd-{uuid.uuid4()}'),
                ('bollinger_bands', 'Bollinger Bands', f'bb-{uuid.uuid4()}'),
            ]:
                if key in additional_figures and additional_figures[key]:
                    chart_data = additional_figures[key].to_json()
                    chart_filename = f"{self.output_dir}/chart_{chart_id}.json"
                    with open(chart_filename, 'w') as f:
                        json.dump(json.loads(chart_data), f, separators=(',', ':'))
                    chart_files['additional_figures'][key] = {'title': title, 'url': chart_filename, 'id': chart_id}

        if predictions and len(historical_data) > 50 and report_type in ["full", "predictions"]:
            pred_days = 7
            prediction_methods = ["technical_analysis", "moving_average", "learning_trend"]
            recent_data = historical_data.tail(20)
            historical_dates = recent_data.index
            historical_prices = recent_data['Close'].values if 'Close' in recent_data.columns else np.zeros(20)
            for method in prediction_methods:
                pred_prices = predictions.predict_prices(pred_days, method=method) if hasattr(predictions, 'predict_prices') else [historical_data['Close'].iloc[-1]] * pred_days
                future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=pred_days, freq='D')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Prices', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=future_dates, y=pred_prices, mode='lines+markers', name=f'Predicted Prices ({method.replace("_", " ").title()})', line=dict(color='red' if method == "technical_analysis" else 'green' if method == "moving_average" else 'purple', dash='dash')))
                fig.update_layout(title=f'7-Day Price Prediction ({method.replace("_", " ").title()})', xaxis_title='Date', yaxis_title='Price ($)', height=400)
                chart_id = f"pred-{method}-{uuid.uuid4()}"
                chart_filename = f"{self.output_dir}/chart_{chart_id}.json"
                with open(chart_filename, 'w') as f:
                    json.dump(json.loads(fig.to_json()), f, separators=(',', ':'))
                chart_files['predictions'][method] = {'title': f'Price Prediction: {method.replace("_", " ").title()}', 'url': chart_filename, 'id': chart_id}
        return chart_files

    def _generate_overview(self, data: pd.DataFrame) -> str:
        """Generate overview section."""
        try:
            if not data.empty:
                start_date = data.index[0].strftime('%Y-%m-%d')
                end_date = data.index[-1].strftime('%Y-%m-%d')
                current_price = data['Close'].iloc[-1] if 'Close' in data.columns else 0
                total_return = ((current_price / data['Close'].iloc[0]) - 1) * 100 if len(data) > 1 and 'Close' in data.columns else 0
                volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100 if len(data) > 1 and 'Close' in data.columns else 0
                return f"""
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricCard title="Period" value="{start_date} to {end_date}" />
                    <MetricCard title="Current Price" value="${current_price:.2f}" />
                    <MetricCard title="Total Return" value="{total_return:.2f}%" />
                    <MetricCard title="Volatility (Annual)" value="{volatility:.2f}%" />
                </div>
                """
            return "<p>No historical data available for overview.</p>"
        except Exception as e:
            return f"<p>Error generating overview: {html.escape(str(e))}</p>"

    def _generate_visualizations(self, chart_files: Dict) -> str:
        """Generate visualizations section with external chart data."""
        html_section = "<p>Interactive charts for advanced analysis.</p>"
        try:
            for key, info in chart_files.items():
                html_section += f"""
                <ChartContainer title="{html.escape(info['title'])}" chartId="{info['id']}" chartUrl="{info['url']}" />
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
                <table className="w-full border-collapse bg-white">
                    <thead>
                        <tr className="bg-gradient-to-r from-primary to-secondary text-white">
                            <th className="p-2 border">Indicator</th>
                            <th className="p-2 border">Signal</th>
                            <th className="p-2 border">Strength</th>
                            <th className="p-2 border">Explanation</th>
                        </tr>
                    </thead>
                    <tbody>
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
                    signal_class = 'text-buy' if 'buy' in signal.lower() else 'text-sell' if 'sell' in signal.lower() else 'text-neutral'
                    html_section += f"""
                    <tr>
                        <td className="p-2 border">{html.escape(indicator)}</td>
                        <td className="{{`p-2 border font-bold ${signal_class}`}}">{html.escape(signal)}</td>
                        <td className="p-2 border">{html.escape(strength)}</td>
                        <td className="p-2 border">{html.escape(explanation)}</td>
                    </tr>
                    """
                html_section += "</tbody></table>"
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
            
            html_section += """
            <h3 className="text-xl font-semibold text-primary mb-2">Signal Summary</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            """
            html_section += f"""
                <MetricCard title="Buy Signals" value="{signal_summary['buy']}" />
                <MetricCard title="Sell Signals" value="{signal_summary['sell']}" />
                <MetricCard title="Hold/Neutral" value="{signal_summary['hold']}" />
            </div>
            """
            
            html_section += "<h3 className='text-xl font-semibold text-primary mb-2'>Overall Recommendation</h3>"
            signal_class = 'text-buy' if signal_summary["buy"] > signal_summary["sell"] else 'text-sell' if signal_summary["sell"] > signal_summary["buy"] else 'text-neutral'
            recommendation = "Bullish Outlook: Consider accumulating or holding positions." if signal_summary["buy"] > signal_summary["sell"] else "Bearish Outlook: Consider reducing positions." if signal_summary["sell"] > signal_summary["buy"] else "Neutral Outlook: Wait for clearer signals."
            html_section += f"<p className='{{`font-bold ${signal_class}`}}'>{recommendation}</p>"
            
            strategies = analytics.generate_trading_strategies(trading_signals) if hasattr(analytics, 'generate_trading_strategies') else []
            if strategies:
                html_section += """
                <details className="mb-4 bg-white rounded-lg shadow-sm">
                    <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg">
                        Recommended Trading Strategies
                    </summary>
                    <div className="p-4">
                """
                for i, strategy in enumerate(strategies):
                    html_section += f"""
                        <h4 className="text-lg font-semibold text-primary">Strategy {i+1}: {html.escape(strategy.get('name', 'Unknown'))}</h4>
                        <p><strong>Type:</strong> {html.escape(strategy.get('type', 'N/A'))}</p>
                        <p><strong>Risk Level:</strong> {html.escape(strategy.get('risk_level', 'N/A'))}</p>
                        <p><strong>Time Horizon:</strong> {html.escape(strategy.get('time_horizon', 'N/A'))}</p>
                        <p><strong>Description:</strong> {html.escape(strategy.get('description', 'N/A'))}</p>
                    """
                html_section += "</div></details>"
            
            risk_metrics = analytics.calculate_risk_metrics() if hasattr(analytics, 'calculate_risk_metrics') else {}
            if risk_metrics:
                html_section += """
                <details className="mb-4 bg-white rounded-lg shadow-sm">
                    <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg">
                        Risk Metrics
                    </summary>
                    <div className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                """
                for metric, value in risk_metrics.items():
                    value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    html_section += f"""
                    <MetricCard title="{html.escape(metric.replace('_', ' ').title())}" value="{html.escape(value_str)}" />
                    """
                html_section += "</div></details>"
            
            patterns = analytics.analyze_patterns() if hasattr(analytics, 'analyze_patterns') else {}
            if patterns:
                html_section += """
                <details className="mb-4 bg-white rounded-lg shadow-sm">
                    <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg">
                        Market Patterns Analysis
                    </summary>
                    <div className="p-4">
                """
                if patterns.get('seasonal_patterns'):
                    html_section += "<h4 className='text-lg font-semibold text-primary'>Seasonal Patterns</h4><ul className='list-disc pl-5'>"
                    for pattern in patterns['seasonal_patterns']:
                        html_section += f"<li>{html.escape(pattern)}</li>"
                    html_section += "</ul>"
                else:
                    html_section += "<p>No significant seasonal patterns detected.</p>"
                if patterns.get('volume_patterns'):
                    html_section += "<h4 className='text-lg font-semibold text-primary'>Volume Patterns</h4><ul className='list-disc pl-5'>"
                    for pattern in patterns['volume_patterns']:
                        html_section += f"<li>{html.escape(pattern)}</li>"
                    html_section += "</ul>"
                else:
                    html_section += "<p>No significant volume patterns detected.</p>"
                html_section += "</div></details>"
        except Exception as e:
            html_section += f"<p>Error generating trading insights: {html.escape(str(e))}</p>"
        return html_section

    def _generate_price_predictions(self, historical_data, predictions, chart_files: Dict) -> str:
        """Generate price predictions section with external chart data."""
        html_section = ""
        try:
            if len(historical_data) > 50:
                pred_days = 7
                prediction_methods = ["technical_analysis", "moving_average", "learning_trend"]
                recent_data = historical_data.tail(20)
                historical_prices = recent_data['Close'].values if 'Close' in recent_data.columns else np.zeros(20)
                future_dates = pd.date_range(start=recent_data.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='D')
                
                html_section += """
                <table className="w-full border-collapse bg-white mb-4">
                    <thead>
                        <tr className="bg-gradient-to-r from-primary to-secondary text-white">
                            <th className="p-2 border">Date</th>
                            <th className="p-2 border">Technical Analysis ($)</th>
                            <th className="p-2 border">Moving Average ($)</th>
                            <th className="p-2 border">Learning Trend ($)</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                pred_prices = {}
                for method in prediction_methods:
                    pred_prices[method] = predictions.predict_prices(pred_days, method=method) if hasattr(predictions, 'predict_prices') else [historical_data['Close'].iloc[-1]] * pred_days
                for i, date in enumerate(future_dates):
                    html_section += f"""
                    <tr>
                        <td className="p-2 border">{date.strftime('%Y-%m-%d')}</td>
                        <td className="p-2 border">{pred_prices['technical_analysis'][i]:.2f}</td>
                        <td className="p-2 border">{pred_prices['moving_average'][i]:.2f}</td>
                        <td className="p-2 border">{pred_prices['learning_trend'][i]:.2f}</td>
                    </tr>
                    """
                html_section += "</tbody></table>"
                
                for method, info in chart_files.items():
                    html_section += f"""
                    <ChartContainer title="{html.escape(info['title'])}" chartId="{info['id']}" chartUrl="{info['url']}" />
                    """
                
                confidence = predictions.calculate_prediction_confidence() if hasattr(predictions, 'calculate_prediction_confidence') else {}
                current_price = historical_data['Close'].iloc[-1] if 'Close' in historical_data.columns else 0
                predicted_final = {method: prices[-1] for method, prices in pred_prices.items()}
                change_pct = {method: ((prices[-1] - current_price) / current_price) * 100 if current_price != 0 else 0 for method, prices in pred_prices.items()}
                
                html_section += """
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                """
                for method in prediction_methods:
                    html_section += f"""
                    <MetricCard title="Predicted Change ({method.replace('_', ' ').title()})" value="{change_pct[method]:.2f}%" />
                    <MetricCard title="Target Price ({method.replace('_', ' ').title()})" value="${predicted_final[method]:.2f}" />
                    """
                html_section += f"""
                    <MetricCard title="Confidence Score" value="{confidence.get('score', 0):.1f}/10" />
                    <MetricCard title="Prediction Volatility" value="{confidence.get('volatility', 0):.2f}%" />
                </div>
                <p className="text-sm italic text-neutral">Disclaimer: Predictions are for informational purposes only.</p>
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
                html_section += """
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                """
                html_section += f"""
                    <MetricCard title="Total Stocks" value="{summary.get('total_stocks', 0)}" />
                    <MetricCard title="Average Change" value="{summary.get('avg_change', 0):.2f}%" />
                    <MetricCard title="Gainers" value="{summary.get('gainers', 0)}" />
                    <MetricCard title="Losers" value="{summary.get('losers', 0)}" />
                </div>
                """
            
            sector_analysis = advanced_analytics.get_sector_analysis() if hasattr(advanced_analytics, 'get_sector_analysis') else pd.DataFrame()
            if not sector_analysis.empty:
                html_section += """
                <details className="mb-4 bg-white rounded-lg shadow-sm">
                    <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg">
                        Sector Performance
                    </summary>
                    <div className="p-4">
                        {sector_analysis.to_html(index=False, border=1, classes="w-full border-collapse bg-white", escape=False)}
                    </div>
                </details>
                """
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
                <details className="mb-4 bg-white rounded-lg shadow-sm">
                    <summary className="cursor-pointer p-3 bg-gradient-to-r from-blue-50 to-purple-50 text-primary hover:bg-gradient-to-l hover:text-secondary rounded-t-lg">
                        Data Summary
                    </summary>
                    <div className="p-4">
                        <table className="w-full border-collapse bg-white">
                            <thead>
                                <tr className="bg-gradient-to-r from-primary to-secondary text-white">
                                    <th className="p-2 border">Metric</th>
                                    <th className="p-2 border">Value</th>
                                </tr>
                            </thead>
                            <tbody>
                """
                for stat, value in summary_stats.items():
                    html_section += f"""
                    <tr>
                        <td className="p-2 border">{html.escape(stat.capitalize())}</td>
                        <td className="p-2 border">{value:.2f}</td>
                    </tr>
                    """
                html_section += "</tbody></table></div></details>"
            else:
                html_section += "<p>No data available for summary.</p>"
        except Exception as e:
            html_section += f"<p>Error generating data summary: {html.escape(str(e))}</p>"
        return html_section

    def save_report_to_file(self, html_content: bytes, filename: Optional[str] = None) -> str:
        """Save HTML report to file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/financial_analysis_report_{timestamp}.html"
        with open(filename, 'wb') as f:
            f.write(html_content)
        return filename
