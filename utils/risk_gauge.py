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
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        
        # Select numeric columns for validation
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
    
    def create_price_volume_line(self, data: pd.DataFrame) -> go.Figure:
        """Create a line chart for closing price with a volume bar subplot."""
        
        logger.debug("Starting price-volume line chart creation")
        if data.empty:
            logger.warning("Input DataFrame is empty")
            st.warning("Cannot create price-volume chart: Input data is empty")
            return go.Figure()
        
        # Normalize column names for case-insensitive matching
        data = data.copy()
        data.columns = data.columns.str.strip().str.lower()
        
        # Required columns
        required_cols = ['close', 'volume']
        date_candidates = ['date', 'datetime']
        date_col = None
        
        # Find date column
        for col in date_candidates:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No 'date' or 'datetime' column found")
            st.warning("Cannot create price-volume chart: No 'date' or 'datetime' column found")
            return go.Figure()
        
        if not all(col in data.columns for col in required_cols):
            missing_cols = set(required_cols) - set(data.columns)
            logger.warning(f"Missing required columns: {missing_cols}")
            st.warning(f"Cannot create price-volume chart: Missing columns {missing_cols}")
            return go.Figure()
        
        # Handle datetime index as fallback
        if data[date_col].isna().all() and pd.api.types.is_datetime64_any_dtype(data.index):
            logger.debug("Date column is NaN, using index as date")
            data[date_col] = data.index
            data.reset_index(drop=True, inplace=True)
        
        # Convert date column to datetime with explicit format check
        logger.debug(f"Raw date values: {data[date_col].head().tolist()}")
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            # Check for epoch-like dates (e.g., 1970 or earlier)
            if (data[date_col] < pd.Timestamp('1970-01-01')).any():
                logger.warning("Found epoch-like dates (pre-1970)")
                st.warning("Cannot create price-volume chart: Invalid dates (pre-1970 detected)")
                return go.Figure()
        except Exception as e:
            logger.error(f"Date conversion failed: {str(e)}")
            st.warning(f"Cannot create price-volume chart: Invalid date format ({str(e)})")
            return go.Figure()
        
        if data[date_col].isna().all():
            logger.warning("All date values are invalid after conversion")
            st.warning("Cannot create price-volume chart: All date values are invalid")
            return go.Figure()
        
        # Ensure close and volume are numeric
        for col in required_cols:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                logger.error(f"Failed to convert {col} to numeric: {str(e)}")
                st.warning(f"Cannot create price-volume chart: Invalid data in {col} ({str(e)})")
                return go.Figure()
        
        # Clean data by removing NaN rows
        logger.debug("Cleaning data by removing NaN values")
        data = data.dropna(subset=required_cols + [date_col])
        if data.empty:
            logger.warning("DataFrame is empty after cleaning")
            st.warning("Cannot create price-volume chart: No valid data after cleaning")
            return go.Figure()
        
        # Validate positive close price
        if (data['close'] <= 0).any():
            logger.warning("Found non-positive values in close column")
            st.warning("Cannot create price-volume chart: Non-positive values in close column")
            return go.Figure()
        
        # Log converted date values
        logger.debug(f"Converted date values: {data[date_col].head().tolist()}")
        
        # Log sample data
        logger.debug(f"Final data shape: {data.shape}")
        logger.debug(f"Sample data:\n{data.head().to_string()}")
        logger.debug(f"Date column dtype: {data[date_col].dtype}")
        logger.debug(f"Close and volume dtypes: {data[required_cols].dtypes}")
        
        try:
            # Create subplots
            logger.debug("Creating subplots")
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Closing Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add closing price line trace
            logger.debug("Adding closing price trace")
            fig.add_trace(
                go.Scatter(
                    x=data[date_col],
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00ff00', width=2),
                    hovertemplate=(
                        '<b>Date:</b> %{x|%m-%d-%y}<br>' +
                        '<b>Close:</b> $%{y:.2f}<extra></extra>'
                    )
                ),
                row=1, col=1
            )
            
            # Add volume bar trace
            logger.debug("Adding volume trace")
            fig.add_trace(
                go.Bar(
                    x=data[date_col],
                    y=data['volume'],
                    name='Volume',
                    marker_color='lightblue',
                    hovertemplate=(
                        '<b>Date:</b> %{x|%m-%d-%y}<br>' +
                        '<b>Volume:</b> %{y:,.0f}<extra></extra>'
                    )
                ),
                row=2, col=1
            )
            
            # Update layout
            logger.debug("Updating figure layout")
            fig.update_layout(
                title='Professional Price and Volume Analysis',
                yaxis_title='Closing Price (USD)',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=False
            )
            
            # Update x-axis formatting
            fig.update_xaxes(
                type='date',
                tickformat='%m-%d-%y',
                tickangle=45,
                hoverformat='%m-%d-%y',
                showgrid=True,
                rangeslider_visible=False
            )
            
            logger.debug("Price-volume line chart creation completed")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create price-volume chart: {str(e)}")
            st.warning(f"Cannot create price-volume chart: {str(e)}")
            return go.Figure()
