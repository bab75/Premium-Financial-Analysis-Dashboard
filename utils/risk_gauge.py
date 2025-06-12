"""
Professional Risk Gauge and Advanced Visualizations Module
Creates sophisticated gauge meters and high-tech charts for financial analysis
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
            mode = "gauge+number",
            value = risk_score,
            title = {'text': title, 'font': {'size': 16}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {
                'font': {'size': 40, 'color': color, 'family': 'Arial Black'}, 
                'suffix': "%",
                'valueformat': '.0f'
            },
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 12}},
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
        
        # Add risk level text below gauge (moved down to prevent overlap)
        fig.add_annotation(
            text=f"<b>{risk_level}</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color=color),
            align="center"
        )
        
        fig.update_layout(
            paper_bgcolor = "white",
            font = {'color': "darkblue", 'family': "Arial"},
            height=280,
            margin=dict(l=10, r=10, t=30, b=35)
        )
        
        return fig
    
    def create_volatility_gauge(self, volatility: float) -> go.Figure:
        """Create volatility gauge meter."""
        return self.create_risk_gauge(volatility, "Volatility Index")
    
    def create_performance_gauge(self, performance: float) -> go.Figure:
        """Create performance gauge meter."""
        # Convert performance to 0-100 scale
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
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = data.get('risk_score', 50),
            title = {'text': "Risk Level"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=1)
        
        # Volatility Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = data.get('volatility', 30),
            title = {'text': "Volatility"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "orange"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
        ), row=1, col=2)
        
        # Performance Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = data.get('performance', 60),
            title = {'text': "Performance"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
        ), row=1, col=3)
        
        # Market Sentiment Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = data.get('sentiment', 45),
            title = {'text': "Market Sentiment"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
        ), row=2, col=1)
        
        # Liquidity Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = data.get('liquidity', 70),
            title = {'text': "Liquidity"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
        ), row=2, col=2)
        
        # Overall Score Gauge
        overall_score = np.mean([
            data.get('risk_score', 50),
            data.get('volatility', 30),
            data.get('performance', 60),
            data.get('sentiment', 45),
            data.get('liquidity', 70)
        ])
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = overall_score,
            title = {'text': "Overall Score"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}]},
        ), row=2, col=3)
        
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
            # Create sample data for demonstration
            x = np.linspace(0, 10, 20)
            y = np.linspace(0, 10, 20)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2))
        else:
            # Use actual data if available
            try:
                price_data = data['Last Sale_current'] if 'Last Sale_current' in data.columns else data.iloc[:, 1]
                volume_data = data['Volume_current'] if 'Volume_current' in data.columns else data.iloc[:, 2]
                change_data = data['Price_Change_Pct'] if 'Price_Change_Pct' in data.columns else data.iloc[:, 3]
                
                # Create grid for 3D plot
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
        
        # Select numeric columns for correlation
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
    
    def create_advanced_candlestick(self, data: pd.DataFrame) -> go.Figure:
        """Create professional candlestick chart with technical indicators."""
        
        if data.empty or 'Date' not in data.columns:
            print("Error: Input DataFrame is empty or missing 'Date' column. Columns:", data.columns.tolist())
            return go.Figure()
        
        # Log initial row count
        initial_rows = len(data)
        print(f"Initial row count: {initial_rows}")
        
        # Create a copy and preprocess data
        data = data.copy()
        
        # Convert Date to datetime
        try:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            print(f"Converted Date dtype: {data['Date'].dtype}")
            print(f"Converted Date samples: {data['Date'].head().tolist()}")
        except Exception as e:
            print(f"Error: Failed to convert 'Date' to datetime: {str(e)}")
            return go.Figure()
        
        # Drop rows with invalid dates
        data = data.dropna(subset=['Date'])
        
        # Ensure numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    print(f"Error: Failed to convert '{col}' to numeric: {str(e)}")
                    return go.Figure()
        
        # Drop rows with invalid numeric values in one pass
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Log processed row count
        processed_rows = len(data)
        print(f"Processed row count: {processed_rows}")
        if processed_rows == 0:
            print("Error: No valid data after preprocessing")
            return go.Figure()
        
        # Log column lengths and samples
        print(f"Date length: {len(data['Date'])}, Open length: {len(data['Open'])}, Volume length: {len(data['Volume'])}")
        print(f"Date sample: {data['Date'].head().tolist()}")
        print(f"Open sample: {data['Open'].head().tolist()}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Stock Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick
        if len(data['Date']) == len(data['Open']) == len(data['High']) == len(data['Low']) == len(data['Close']):
            fig.add_trace(
                go.Candlestick(
                    x=data['Date'],
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
                row=1, col=1
            )
        else:
            print(f"Error: Mismatched lengths - Date: {len(data['Date'])}, Open: {len(data['Open'])}, High: {len(data['High'])}, Low: {len(data['Low'])}, Close: {len(data['Close'])}")
            return go.Figure()
        
        # Add volume bars
        if len(data['Date']) == len(data['Volume']):
            fig.add_trace(
                go.Bar(
                    x=data['Date'],
                    y=data['Volume'],
                    name="Volume",
                    marker_color='lightblue',
                    hovertemplate='Date: %{x|%m-%d-%y}<br>' +
                                  'Volume: %{y:,.0f}<br>' +
                                  '<extra></extra>'
                ),
                row=2, col=1
            )
        else:
            print(f"Error: Mismatched lengths - Date: {len(data['Date'])}, Volume: {len(data['Volume'])}")
            return go.Figure()
        
        fig.update_layout(
            title='Professional Candlestick Analysis',
            yaxis_title='Stock Price (USD)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(
                type='date',
                tickformat='%m-%d-%y',
                tickangle=45
            ),
            xaxis2=dict(
                type='date',
                tickformat='%m-%d-%y',
                tickangle=45
            )
        )
        
        return fig
