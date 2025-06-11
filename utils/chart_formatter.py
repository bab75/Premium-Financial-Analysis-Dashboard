"""
Universal Chart Formatter with Date Handling and Explanations
Ensures consistent MM-DD-YY date formatting across all charts
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any

class ChartFormatter:
    """Universal chart formatter for consistent date display and explanations."""
    
    def __init__(self):
        self.explanations = {
            'risk_gauge': {
                'title': 'Risk Assessment Gauge',
                'explanation': '''This professional gauge meter displays your portfolio's risk level on a scale of 0-100%.
                
**How to Read:**
- 0-25% (Green): Low Risk - Conservative investments with stable returns
- 25-50% (Yellow): Medium Risk - Balanced portfolio with moderate volatility
- 50-75% (Orange): High Risk - Growth-focused with higher volatility
- 75-100% (Red): Extreme Risk - Aggressive investments with high potential gains/losses

**Real-time Example:** If your gauge shows 35%, your portfolio has medium risk, suggesting a balanced mix of conservative and growth investments suitable for moderate risk tolerance.''',
                'use_case': 'Use this to quickly assess if your portfolio matches your risk tolerance and investment goals.'
            },
            
            'volatility_gauge': {
                'title': 'Volatility Index',
                'explanation': '''Measures price fluctuation intensity over time. Higher values indicate more price swings.
                
**Volatility Levels:**
- 0-20%: Very stable, minimal price changes
- 20-40%: Moderate fluctuations, normal market behavior
- 40-60%: High volatility, significant price swings
- 60%+: Extreme volatility, major market events

**Real-time Example:** A volatility of 45% means the stock price typically moves 45% up or down from its average, indicating active trading and price sensitivity to news.''',
                'use_case': 'Higher volatility = higher potential returns but also higher risk of losses.'
            },
            
            'performance_gauge': {
                'title': 'Performance Score',
                'explanation': '''Composite score measuring overall investment performance across multiple factors.
                
**Score Breakdown:**
- 0-25%: Poor performance, consider reviewing strategy
- 25-50%: Below average, room for improvement
- 50-75%: Good performance, meeting expectations
- 75-100%: Excellent performance, outperforming market

**Real-time Example:** A score of 72% indicates strong performance, meaning your investments are performing better than average market conditions.''',
                'use_case': 'Use this to evaluate overall investment success and compare against benchmarks.'
            },
            
            '3d_factor_analysis': {
                'title': '3D Factor Analysis',
                'explanation': '''Advanced visualization showing the relationship between three key financial metrics simultaneously.
                
**Three Dimensions Explained:**
- X-Axis: Risk Level (volatility measure)
- Y-Axis: Return Potential (expected gains)
- Z-Axis: Market Correlation (how closely it follows market trends)
- Color: Performance Score (overall success rating)

**How to Interpret:**
- High peaks: Stocks with strong performance across all factors
- Clusters: Groups of stocks with similar characteristics
- Outliers: Unique opportunities or risks requiring attention

**Real-time Example:** A peak at coordinates (30, 80, 60) represents a stock with medium risk (30%), high return potential (80%), and moderate market correlation (60%).''',
                'use_case': 'Identify optimal investment opportunities by finding stocks that balance risk, return, and market behavior according to your preferences.'
            },
            
            'moving_averages': {
                'title': 'Moving Averages Chart',
                'explanation': '''Shows price trends using Simple Moving Averages (SMA) to smooth out price fluctuations.
                
**Key Lines:**
- Blue: Current stock price (actual daily closing prices)
- Green: 20-day SMA (short-term trend)
- Red: 50-day SMA (medium-term trend)

**Trading Signals:**
- Price above both SMAs: Upward trend (bullish)
- Price below both SMAs: Downward trend (bearish)
- SMA crossovers: Potential trend changes

**Real-time Example:** When the green line (20-day) crosses above the red line (50-day), it often signals the start of an upward price trend.''',
                'use_case': 'Use to identify trend direction and optimal entry/exit points for trading decisions.'
            },
            
            'rsi_chart': {
                'title': 'Relative Strength Index (RSI)',
                'explanation': '''Momentum oscillator measuring speed and magnitude of price changes on a 0-100 scale.
                
**Key Zones:**
- Above 70: Overbought (price may decrease soon)
- Below 30: Oversold (price may increase soon)
- Around 50: Neutral momentum

**Trading Signals:**
- RSI crossing above 30: Potential buying opportunity
- RSI crossing below 70: Potential selling opportunity
- Divergences: Price and RSI moving in opposite directions

**Real-time Example:** If RSI drops to 25, the stock is oversold and historically likely to rebound, presenting a potential buying opportunity.''',
                'use_case': 'Identify overbought/oversold conditions to time your buy and sell decisions more effectively.'
            },
            
            'bollinger_bands': {
                'title': 'Bollinger Bands',
                'explanation': '''Price channel showing volatility and potential support/resistance levels.
                
**Components:**
- Middle Band: 20-day moving average (trend direction)
- Upper Band: Resistance level (price ceiling)
- Lower Band: Support level (price floor)
- Band Width: Indicates volatility level

**Trading Signals:**
- Price touches upper band: Potential resistance, consider selling
- Price touches lower band: Potential support, consider buying
- Band squeeze: Low volatility, expecting price breakout

**Real-time Example:** When price consistently bounces between $45 (lower band) and $55 (upper band), you can buy near $45 and sell near $55.''',
                'use_case': 'Identify optimal entry and exit points based on volatility-adjusted support and resistance levels.'
            }
        }
    
    def format_date_axis(self, fig: go.Figure) -> go.Figure:
        """Apply consistent MM-DD-YY date formatting to chart x-axis."""
        fig.update_layout(
            xaxis=dict(
                tickformat='%m-%d-%y',
                tickmode='auto',
                nticks=10,
                tickangle=45
            )
        )
        return fig
    
    def add_explanation_section(self, chart_type: str) -> Dict[str, str]:
        """Get explanation content for a specific chart type."""
        return self.explanations.get(chart_type, {
            'title': 'Financial Chart',
            'explanation': 'This chart displays financial data for analysis.',
            'use_case': 'Use this chart to analyze financial trends and patterns.'
        })
    
    def format_chart_with_explanation(self, fig: go.Figure, chart_type: str, 
                                    additional_info: str = "") -> tuple:
        """Format chart and return with explanation."""
        # Apply date formatting
        formatted_fig = self.format_date_axis(fig)
        
        # Get explanation
        explanation = self.add_explanation_section(chart_type)
        
        if additional_info:
            explanation['explanation'] += f"\n\n**Additional Information:** {additional_info}"
        
        return formatted_fig, explanation
    
    def ensure_date_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has proper datetime index for chart formatting."""
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                data.index = pd.to_datetime(data.index)
            except:
                # If conversion fails, create a date range
                data.index = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
        
        return data