import plotly.io as pio
from datetime import datetime as dt
import pandas as pd
import numpy as np

class HTMLReportGenerator:
    def generate_comprehensive_report(
        self,
        stock_symbol,
        historical_data,
        tech_indicators,
        analytics,
        visualizations,
        predictions,
        advanced_analytics,
        additional_figures=None,
        report_type="full"
    ):
        if additional_figures is None:
            additional_figures = {}

        html_content = f"""
        <html>
        <head>
            <title>Financial Analysis Report - {stock_symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2, h3 {{ color: #34495e; }}
                .chart-container {{ margin: 20px 0; padding: 10px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .section {{ margin: 30px 0; }}
                .metric-card {{ display: flex; flex-wrap: wrap; gap: 10px; }}
                .metric {{ background-color: #e8ecef; padding: 10px; border-radius: 5px; flex: 1; min-width: 150px; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .disclaimer {{ font-style: italic; color: #7f8c8d; font-size: 0.9em; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Financial Analysis Report: {stock_symbol}</h1>
            <p style="text-align: center;">Generated on: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        html_content += """
        <div class="section">
            <h2>Overview</h2>
        """
        if not historical_data.empty:
            start_date = pd.to_datetime(historical_data.index[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(historical_data.index[-1]).strftime('%Y-%m-%d')
            current_price = historical_data['Close'].iloc[-1]
            total_return = ((current_price / historical_data['Close'].iloc[0]) - 1) * 100
            volatility = historical_data['Close'].pct_change().std() * np.sqrt(252) * 100
            html_content += f"""
            <div class="metric-card">
                <div class="metric"><strong>Period</strong><br>{start_date} to {end_date}</div>
                <div class="metric"><strong>Current Price</strong><br>${current_price:.2f}</div>
                <div class="metric"><strong>Total Return</strong><br>{total_return:.2f}%</div>
                <div class="metric"><strong>Volatility (Annual)</strong><br>{volatility:.2f}%</div>
            </div>
            """

        html_content += """
        <div class="section">
            <h2>Advanced Visualizations</h2>
        """

        def add_chart(fig, title, chart_id):
            if fig and hasattr(fig, 'data') and fig.data:
                chart_html = pio.to_html(fig, full_html=False, div_id=chart_id)
                return f"""
                <div class="chart-container">
                    <h3>{title}</h3>
                    {chart_html}
                </div>
                """
            return ""

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

        for key, title, chart_id in chart_mappings:
            if key in additional_figures:
                html_content += add_chart(additional_figures[key], title, chart_id)

        html_content += """
        <div class="section">
            <h2>Technical Indicators</h2>
        """
        try:
            signals = tech_indicators.get_trading_signals()
            if signals:
                html_content += "<h3>Trading Signals</h3><table><tr><th>Indicator</th><th>Signal</th><th>Strength</th></tr>"
                for indicator, signal_data in signals.items():
                    signal = signal_data.get('signal', 'Unknown')
                    strength = signal_data.get('strength', 'Unknown')
                    html_content += f"<tr><td>{indicator}</td><td>{signal}</td><td>{strength}</td></tr>"
                html_content += "</table>"
        except Exception as e:
            html_content += f"<p>Error generating trading signals: {str(e)}</p>"

        html_content += """
        <div class="section">
            <h2>Trading Insights</h2>
        """
        try:
            trading_signals = tech_indicators.get_trading_signals()
            signal_summary = {"buy": 0, "sell": 0, "hold": 0}
            signal_details = []
            for indicator, signal_data in trading_signals.items():
                signal = signal_data.get('signal', 'Unknown').lower()
                strength = signal_data.get('strength', 'Unknown')
                details = f"Indicator: {indicator}, Signal: {signal_data.get('signal', 'Unknown')}, Strength: {strength}"
                if indicator == 'RSI':
                    rsi = tech_indicators.calculate_rsi().iloc[-1] if len(tech_indicators.calculate_rsi()) > 0 else 0
                    details += f", Current RSI: {rsi:.1f}"
                    if rsi > 70:
                        details += " (Overbought)"
                    elif rsi < 30:
                        details += " (Oversold)"
                elif indicator == 'MACD':
                    macd = tech_indicators.calculate_macd()
                    if not macd.empty:
                        details += f", MACD Diff: {(macd['MACD'] - macd['Signal']).iloc[-1]:.2f}"
                signal_details.append(details)
                if 'buy' in signal:
                    signal_summary["buy"] += 1
                elif 'sell' in signal:
                    signal_summary["sell"] += 1
                else:
                    signal_summary["hold"] += 1
            html_content += f"""
            <div class="metric-card">
                <div class="metric"><strong>Buy Signals</strong><br>{signal_summary['buy']}</div>
                <div class="metric"><strong>Sell Signals</strong><br>{signal_summary['sell']}</div>
                <div class="metric"><strong>Hold/Neutral</strong><br>{signal_summary['hold']}</div>
            </div>
            <h3>Signal Details</h3>
            <ul>
            """
            for detail in signal_details:
                html_content += f"<li>{detail}</li>"
            html_content += "</ul>"

            strategies = analytics.generate_trading_strategies(trading_signals)
            if strategies:
                html_content += "<h3>Recommended Strategies</h3>"
                for i, strategy in enumerate(strategies):
                    html_content += f"""
                    <div class="chart-container">
                        <h4>Strategy {i+1}: {strategy.get('name', 'Unknown')}</h4>
                        <p><strong>Type:</strong> {strategy.get('type', 'N/A')}</p>
                        <p><strong>Risk Level:</strong> {strategy.get('risk_level', 'N/A')}</p>
                        <p><strong>Description:</strong> {strategy.get('description', 'N/A')}</p>
                    </div>
                    """
            risk_metrics = analytics.calculate_risk_metrics()
            html_content += f"""
            <h3>Risk Metrics</h3>
            <div class="metric-card">
                <div class="metric"><strong>Beta</strong><br>{risk_metrics.get('beta', 'N/A')}</div>
                <div class="metric"><strong>Sharpe Ratio</strong><br>{risk_metrics.get('sharpe_ratio', 'N/A')}</div>
                <div class="metric"><strong>Max Drawdown</strong><br>{risk_metrics.get('max_drawdown', 'N/A')}%</div>
                <div class="metric"><strong>VaR (95%)</strong><br>{risk_metrics.get('var_95', 'N/A')}%</div>
            </div>
            """
            patterns = analytics.analyze_patterns()
            html_content += "<h3>Pattern Analysis</h3>"
            html_content += f"""
            <p><strong>Seasonal Patterns:</strong> {', '.join(patterns.get('seasonal_patterns', ['None']))}</p>
            <p><strong>Volume Patterns:</strong> {', '.join(patterns.get('volume_patterns', ['None']))}</p>
            """
        except Exception as e:
            html_content += f"<p>Error generating trading insights: {str(e)}</p>"

        html_content += """
        <div class="section disclaimer">
            <p><strong>Disclaimer:</strong> This report is for informational purposes only.</p>
        </div>
        </body>
        </html>
        """
        return html_content.encode('utf-8')
