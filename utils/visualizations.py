import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

class TechnicalIndicators:
    """Calculate and visualize technical indicators for stock data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#78909c',
            'signal': 'orange'
        }
        
        # Ensure Datetime column is in datetime format
        if 'Datetime' in self.data.columns:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
        elif 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.rename(columns={'Date': 'Datetime'})
        else:
            # If no Datetime column, assume index is datetime
            self.data = self.data.reset_index()
            if 'index' in self.data.columns:
                self.data = self.data.rename(columns={'index': 'Datetime'})
                self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
    
    def calculate_sma(self, window: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        return self.data['Close'].rolling(window=window, min_periods=1).mean()
    
    def calculate_ema(self, window: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return self.data['Close'].ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal Line, and Histogram."""
        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(window)
        std = self.data['Close'].rolling(window=window, min_periods=1).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return sma, upper_band, lower_band
    
    def get_bollinger_position(self) -> str:
        """Determine price position relative to Bollinger Bands."""
        sma, upper, lower = self.calculate_bollinger_bands()
        current_price = self.data['Close'].iloc[-1]
        if current_price > upper.iloc[-1]:
            return "Above Upper Band"
        elif current_price < lower.iloc[-1]:
            return "Below Lower Band"
        else:
            return "Within Bands"
    
    def get_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals based on indicators."""
        signals = {}
        
        # RSI Signal
        rsi = self.calculate_rsi()
        if rsi.iloc[-1] > 70:
            signals['RSI'] = {'signal': 'Sell', 'strength': 'Strong'}
        elif rsi.iloc[-1] < 30:
            signals['RSI'] = {'signal': 'Buy', 'strength': 'Strong'}
        else:
            signals['RSI'] = {'signal': 'Hold', 'strength': 'Neutral'}
        
        # MACD Signal
        macd, signal, _ = self.calculate_macd()
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals['MACD'] = {'signal': 'Buy', 'strength': 'Moderate'}
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals['MACD'] = {'signal': 'Sell', 'strength': 'Moderate'}
        else:
            signals['MACD'] = {'signal': 'Hold', 'strength': 'Neutral'}
        
        # Bollinger Bands Signal
        bb_position = self.get_bollinger_position()
        if bb_position == "Above Upper Band":
            signals['Bollinger Bands'] = {'signal': 'Sell', 'strength': 'Strong'}
        elif bb_position == "Below Lower Band":
            signals['Bollinger Bands'] = {'signal': 'Buy', 'strength': 'Strong'}
        else:
            signals['Bollinger Bands'] = {'signal': 'Hold', 'strength': 'Neutral'}
        
        return signals
    
    def create_moving_averages_chart(self) -> go.Figure:
        """Create chart showing SMA and EMA with price."""
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # SMA
        sma = self.calculate_sma(window=20)
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=sma,
            mode='lines',
            name='SMA (20)',
            line=dict(color=self.colors['neutral'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'SMA (20): $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # EMA
        ema = self.calculate_ema(window=20)
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=ema,
            mode='lines',
            name='EMA (20)',
            line=dict(color=self.colors['bearish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'EMA (20): $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Moving Averages Analysis',
            xaxis_title='Time Period',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(
                type='date',
                showticklabels=False,  # Consistent with visualizations.py
                showgrid=True,
                hoverformat='%m-%d-%y'
            )
        )
        
        return fig
    
    def create_rsi_chart(self) -> go.Figure:
        """Create RSI chart with overbought/oversold levels."""
        fig = go.Figure()
        
        # RSI
        rsi = self.calculate_rsi()
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=rsi,
            mode='lines',
            name='RSI (14)',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'RSI: %{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color=self.colors['bearish'], annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color=self.colors['bullish'], annotation_text="Oversold")
        
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Time Period',
            yaxis_title='RSI',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(
                type='date',
                showticklabels=False,
                showgrid=True,
                hoverformat='%m-%d-%y'
            ),
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_macd_chart(self) -> go.Figure:
        """Create MACD chart with signal line and histogram."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            subplot_titles=['MACD', 'Histogram']
        )
        
        # MACD and Signal Line
        macd, signal, histogram = self.calculate_macd()
        
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=macd,
            mode='lines',
            name='MACD',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'MACD: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=signal,
            mode='lines',
            name='Signal Line',
            line=dict(color=self.colors['signal'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Signal: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=1, col=1)
        
        # Histogram
        fig.add_trace(go.Bar(
            x=self.data['Datetime'],
            y=histogram,
            name='Histogram',
            marker_color=[self.colors['bullish'] if val > 0 else self.colors['bearish'] for val in histogram],
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Histogram: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=2, col=1)
        
        fig.update_layout(
            title='MACD Analysis',
            hovermode='x unified',
            showlegend=True,
            xaxis2=dict(
                type='date',
                showticklabels=False,
                showgrid=True,
                hoverformat='%m-%d-%y'
            )
        )
        
        fig.update_yaxes(title_text='MACD', row=1, col=1)
        fig.update_yaxes(title_text='Histogram', row=2, col=1)
        fig.update_xaxes(title_text='Time Period', row=2, col=1)
        
        return fig
    
    def create_bollinger_bands_chart(self) -> go.Figure:
        """Create Bollinger Bands chart with price."""
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Bollinger Bands
        sma, upper, lower = self.calculate_bollinger_bands()
        
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=upper,
            mode='lines',
            name='Upper Band',
            line=dict(color=self.colors['neutral'], width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Upper Band: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=lower,
            mode='lines',
            name='Lower Band',
            line=dict(color=self.colors['neutral'], width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Lower Band: $%{y:.2f}<br>' +
                          '<extra></extra>',
            fill='tonexty',
            fillcolor='rgba(120, 144, 156, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data['Datetime'],
            y=sma,
            mode='lines',
            name='SMA (20)',
            line=dict(color=self.colors['signal'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'SMA (20): $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Bollinger Bands Analysis',
            xaxis_title='Time Period',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(
                type='date',
                showticklabels=False,
                showgrid=True,
                hoverformat='%m-%d-%y'
            )
        )
        
        return fig
