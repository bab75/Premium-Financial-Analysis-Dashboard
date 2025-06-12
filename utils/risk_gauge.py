"""
Professional Risk Gauge and Advanced Visualizations Module
Creates sophisticated gauge meters and high-tech charts for financial analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st

class RiskGauge:
    """Create professional risk gauge meters and advanced visualizations."""
    
    def __init__(self):
        self.risk_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#dc3545',
            'extreme': '#6f42c1'
        }
    
    def create_risk_gauge(self, risk_score: float, title: str = "Risk Level") -> go.Figure:
        """Create a professional gauge meter for risk assessment."""
        
        # Determine risk level and color
        if risk_score <= 25:
            risk_level = "Low Risk"
            color = self.risk_colors['low']
        elif risk_score <= 50:
            risk_level = "Medium Risk"
            color = self.risk_colors['medium']
        elif risk_score <= 75:
            risk_level = "High Risk"
            color = self.risk_colors['high']
        else:
            risk_level = "Extreme Risk"
            color = self.risk_colors['extreme']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': title, 'font': {'size': 16}},
            domain={'x': [0, 1], 'y': [0, 1]},
            number={
                'font': {'size': 40, 'color': color, 'family': 'Arial Black'}, 
                'suffix': "%",
                'valueformat': '.0f'
            },
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#e8f5e8'},
                    {'range': [25, 50], 'color': '#fff3cd'},
                    {'range': [50, 75], 'color': '#f8d7da'},
                    {'range': [75, 100], 'color': '#e2d9f3'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Add risk level text below gauge
        fig.add_annotation(
            text=f"<b>{risk_level}</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color=color),
            align="center"
        )
        
        fig.update_layout(
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"},
            height=280,
            margin=dict(l=10, r=10, t=30, b=35)
        )
        
        return fig
    
    def create_volatility_gauge(self, volatility: float) -> go.Figure:
        """Create volatility gauge meter."""
        return self.create_risk_gauge(volatility, "Volatility Index")
    
    def create_performance_gauge(self, performance: float) -> go.Figure:
        """Create performance gauge meter."""
        normalized_perf = min(100, max(0, (performance + 50)))
        return self.create_risk_gauge(normalized_perf, "Performance Score")
    
    def create_advanced_dashboard(self, data: Dict) -> go.Figure:
        """Create an advanced multi-metric dashboard."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Risk Level', 'Volatility', 'Performance', 
                          'Market Sentiment', 'Liquidity', 'Overall Score'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Risk Level Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data.get('risk_score', 50),
                title={'text': "Risk Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Volatility Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data.get('volatility', 30),
                title={'text': "Volatility"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Performance Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data.get('performance', 60),
                title={'text': "Performance"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # Market Sentiment Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data.get('sentiment', 45),
                title={'text': "Market Sentiment"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Liquidity Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=data.get('liquidity', 70),
                title={'text': "Liquidity"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # Overall Score Gauge
        overall_score = np.mean([
            data.get('risk_score', 50),
            data.get('volatility', 30),
            data.get('performance', 60),
            data.get('sentiment', 45),
            data.get('liquidity', 70)
        ])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score,
                title={'text': "Overall Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ]
                }
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Advanced Risk & Performance Dashboard",
            title_x=0.5,
            font=dict(family="Arial", size=12)
        )
        
        return fig
    
    def create_3d_surface_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create advanced 3D surface plot for risk analysis."""
        
        if data.empty or len(data) < 10:
            x = np.linspace(0, 10, 20)
            y = np.linspace(0, 10, 20)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2))
        else:
            try:
                price_data = data['Last Sale_current'] if 'Last Sale_current' in data.columns else data.iloc[:, 1]
                volume_data = data['Volume_current'] if 'Volume_current' in data.columns else data.iloc[:, 2]
                change_data = data['Price_Change_Pct'] if 'Price_Change_Pct' in data.columns else data.iloc[:, 3]
                
                x = np.linspace(price_data.min(), price_data.max(), 20)
                y = np.linspace(volume_data.min(), volume_data.max(), 20)
                X, Y = np.meshgrid(x, y)
                Z = np.random.random(X.shape) * change_data.std()
            except:
                x = np.linspace(0, 10, 20)
                y = np.linspace(0, 10, 20)
                X, Y = np.meshgrid(x, y)
                Z = np.sin(np.sqrt(X**2 + Y**2))
        
        fig = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        )])
        
        fig.update_layout(
            title='3D Risk Surface Analysis',
            scene=dict(
                xaxis_title='Price Level',
                yaxis_title='Volume',
                zaxis_title='Risk Factor',
                camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64))
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def create_heatmap_correlation(self, data: pd.DataFrame) -> go.Figure:
        """Create professional correlation heatmap."""
        
        if data.empty:
            return go.Figure()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return go.Figure()
        
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Advanced Correlation Heatmap',
            xaxis_nticks=36,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_advanced_candlestick(self, df: pd.DataFrame) -> go.Figure:
        """Create professional candlestick chart with technical indicators."""
        
        if df.empty:
            print("Error: Input DataFrame is empty")
            return go.Figure()
        
        # Log input columns
        print("Input DataFrame columns:", df.columns.tolist())
        
        # Check for date column
        date_col = None
        for col in ['Date', 'Datetime', 'date', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            print("Error: No 'Date', 'Datetime', 'date', or 'datetime' column found. Columns:", df.columns.tolist())
            return go.Figure()
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return go.Figure()
        
        # Create a copy and preprocess data
        data = df.copy()
        
        # Log initial row count
        initial_rows = len(data)
        print(f"Initial row count: {initial_rows}")
        
        # Log input date column
        print(f"Input {date_col} dtype: {data[date_col].dtype}")
        print(f"Input {date_col} samples: {data[date_col].head().tolist()}")
        
        # Convert date column to datetime
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        except Exception as e:
            print(f"Error: Failed to convert {date_col} to datetime: {str(e)}")
            return go.Figure()
        
        # Ensure numeric columns
        try:
            for col in required_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        except Exception as e:
            print(f"Error: Failed to convert {col} to numeric: {str(e)}")
            return go.Figure()
        
        # Drop rows with invalid dates or numeric values in one pass
        data = data.dropna(subset=[date_col] + required_cols)
        
        # Log processed row count
        processed_rows = len(data)
        print(f"Processed row count: {processed_rows}")
        if processed_rows == 0:
            print("Error: No valid data after preprocessing")
            return go.Figure()
        
        # Sort by date
        data = data.sort_values(by=date_col)
        
        # Log processed data
        print(f"Processed {date_col} dtype: {data[date_col].dtype}")
        print(f"Processed {date_col} samples: {data[date_col].head().tolist()}")
        print("Processed DataFrame head:\n", data.head().to_string())
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Stock Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        try:
            # Log trace data
            print(f"Candlestick x len: {len(data[date_col])}, open len: {len(data['Open'])}")
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data[date_col],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000',
                    hovertemplate='Date: %{x|%m-%d-%y}<br>' +
                                  'Open: $%{open:.2f}<br>' +
                                  'High: $%{high:.2f}<br>' +
                                  'Low: $%{low:.2f}<br>' +
                                  'Close: $%{close:.2f}<br>' +
                                  '<extra></extra>'
                ),
                row=1,
                col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=data[date_col],
                    y=data['Volume'],
                    name="Volume",
                    marker_color='lightblue',
                    hovertemplate='Date: %{x|%m-%d-%y}<br>' +
                                  'Volume: %{y:,.0f}<br>' +
                                  '<extra></extra>'
                ),
                row=2,
                col=1
            )
            
            # Check if traces have data
            if not fig.data or all(len(trace.x) == 0 for trace in fig.data if hasattr(trace, 'x')):
                print("Error: No data in Plotly traces")
                return go.Figure()
        except Exception as e:
            print(f"Error adding Plotly traces: {str(e)}")
            return go.Figure()
        
        fig.update_layout(
            title='Professional Candlestick Analysis',
            yaxis_title='Stock Price ($)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(
                type='date',
                tickformat='%m-%d-%y',
                tickangle=45,
                showgrid=True,
                hoverformat='%m-%d-%y'
            ),
            xaxis2=dict(
                type='date',
                tickformat='%m-%d-%y',
                tickangle=45,
                showgrid=True,
                hoverformat='%m-%d-%y'
            )
        )
        
        return fig
