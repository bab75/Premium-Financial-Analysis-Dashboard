"""
Universal Chart Fixes for Date Display and Hover Text
Ensures all charts properly display dates in MM-DD-YY format with proper hover text
"""

import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Union

def add_date_hover_template(trace_type: str = "scatter") -> str:
    """Generate proper hover template with MM-DD-YY date format."""
    if trace_type == "candlestick":
        return ('<b>Date:</b> %{x|%m-%d-%y}<br>' +
                '<b>Open:</b> $%{open:.2f}<br>' +
                '<b>High:</b> $%{high:.2f}<br>' +
                '<b>Low:</b> $%{low:.2f}<br>' +
                '<b>Close:</b> $%{close:.2f}<br>' +
                '<extra></extra>')
    elif trace_type == "price":
        return ('<b>Date:</b> %{x|%m-%d-%y}<br>' +
                '<b>Price:</b> $%{y:.2f}<br>' +
                '<extra></extra>')
    elif trace_type == "volume":
        return ('<b>Date:</b> %{x|%m-%d-%y}<br>' +
                '<b>Volume:</b> %{y:,.0f}<br>' +
                '<extra></extra>')
    elif trace_type == "indicator":
        return ('<b>Date:</b> %{x|%m-%d-%y}<br>' +
                '<b>Value:</b> %{y:.2f}<br>' +
                '<extra></extra>')
    else:
        return ('<b>Date:</b> %{x|%m-%d-%y}<br>' +
                '<b>Value:</b> %{y:.2f}<br>' +
                '<extra></extra>')

def fix_chart_dates(fig: go.Figure) -> go.Figure:
    """Apply consistent date formatting to chart x-axis."""
    fig.update_layout(
        xaxis=dict(
            tickformat='%m-%d-%y',
            tickmode='auto',
            nticks=10,
            tickangle=45
        )
    )
    return fig

def ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure data has proper datetime index for charts."""
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        elif 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.set_index('Datetime')
        else:
            # Create date range if no date column found
            data.index = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
    
    return data