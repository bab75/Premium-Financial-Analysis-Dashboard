import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

class TechnicalIndicators:
    """Calculate and visualize technical indicators for stock analysis."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data = self._prepare_data()
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for technical analysis."""
        # Ensure required columns
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Debug: Log input data structure
        print("Input DataFrame columns:", self.data.columns.tolist())
        print("Input DataFrame head:\n", self.data.head().to_string())
        
        # Handle datetime
        if 'Datetime' in self.data.columns:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], errors='coerce')
        elif 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            self.data = self.data.rename(columns={'Date': 'Datetime'})
        elif pd.api.types.is_datetime64_any_dtype(self.data.index):
            self.data = self.data.reset_index()
            if 'index' in self.data.columns:
                self.data = self.data.rename(columns={'index': 'Datetime'})
            elif 'Date' in self.data.columns:
                self.data = self.data.rename(columns={'Date': 'Datetime'})
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], errors='coerce')
        else:
            raise ValueError("No valid Datetime or Date column/index found in data")
        
        # Check for invalid Datetime values
        if self.data['Datetime'].isna().any():
            print("Warning: NaT values found in Datetime column")
            self.data = self.data.dropna(subset=['Datetime'])
        
        # Sort by Datetime
        self.data = self.data.sort_values('Datetime')
        
        # Debug: Log prepared data
        print("Prepared DataFrame columns:", self.data.columns.tolist())
        print("Prepared Datetime head:", self.data['Datetime'].head().to_string())
        
        return self.data
    
    def calculate_sma(self, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return self.data['Close'].rolling(window=period, min_periods=1).mean()
    
    def calculate_ema(self, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return self.data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(period)
        std = self.data['Close'].rolling(window=period, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'Middle': sma,
            'Upper': upper_band,
            'Lower': lower_band
        }
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = self.data['Low'].rolling(window=k_period, min_periods=1).min()
        highest_high = self.data['High'].rolling(window=k_period, min_periods=1).max()
        
        k_percent = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators and return combined DataFrame."""
        indicators = self.data.copy()
        
        # Moving Averages
        indicators['SMA_10'] = self.calculate_sma(10)
        indicators['SMA_20'] = self.calculate_sma(20)
        indicators['SMA_50'] = self.calculate_sma(50)
        indicators['EMA_10'] = self.calculate_ema(10)
        indicators['EMA_20'] = self.calculate_ema(20)
        indicators['EMA_50'] = self.calculate_ema(50)
        
        # RSI
        indicators['RSI'] = self.calculate_rsi()
        
        # MACD
        macd_data = self.calculate_macd()
        indicators['MACD'] = macd_data['MACD']
        indicators['MACD_Signal'] = macd_data['Signal']
        indicators['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands()
        indicators['BB_Upper'] = bb_data['Upper']
        indicators['BB_Middle'] = bb_data['Middle']
        indicators['BB_Lower'] = bb_data['Lower']
        
        # Stochastic
        stoch_data = self.calculate_stochastic()
        indicators['Stoch_K'] = stoch_data['K']
        indicators['Stoch_D'] = stoch_data['D']
        
        return indicators
    
    def create_moving_averages_chart(self) -> go.Figure:
        """Create moving averages chart."""
        fig = go.Figure()
        
        x_data = self.data['Datetime'] if 'Datetime' in self.data.columns else self.data.index
        
        # Price line
        fig.add_trace(go.Scatter(
            x=x_data,
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Moving Averages
        fig.add_trace(go.Scatter(
            x=x_data,
            y=self.calculate_sma(10),
            mode='lines',
            name='SMA 10',
            line=dict(color='#ff7f0e', width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'SMA 10: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=self.calculate_sma(20),
            mode='lines',
            name='SMA 20',
            line=dict(color='#2ca02c', width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'SMA 20: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=self.calculate_sma(50),
            mode='lines',
            name='SMA 50',
            line=dict(color='#d62728', width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'SMA 50: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Price with Moving Averages',
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
    
    def create_rsi_chart(self) -> go.Figure:
        """Create RSI chart."""
        rsi = self.calculate_rsi()
        
        fig = go.Figure()
        
        x_data = self.data['Datetime'] if 'Datetime' in self.data.columns else self.data.index
        
        # RSI Line
        fig.add_trace(go.Scatter(
            x=x_data,
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='#9467bd', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'RSI: %{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Overbought/Oversold Lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
        
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Time Period',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            xaxis=dict(
                type='date',
                showticklabels=False,
                showgrid=True,
                hoverformat='%m-%d-%y'
            )
        )
        
        return fig
    
    def create_macd_chart(self) -> go.Figure:
        """Create MACD chart."""
        macd_data = self.calculate_macd()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
        x_data = self.data['Datetime'] if 'Datetime' in self.data.columns else self.data.index
        
        # MACD and Signal Lines
        fig.add_trace(go.Scatter(
            x=x_data,
            y=macd_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'MACD: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=macd_data['Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Signal: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=1, col=1)
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in macd_data['Histogram']]
        fig.add_trace(go.Bar(
            x=x_data,
            y=macd_data['Histogram'],
            name='Histogram',
            marker_color=colors,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Histogram: %{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=2, col=1)
        
        fig.update_layout(
            title='MACD (Moving Average Convergence Divergence)',
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
        """Create Bollinger Bands chart."""
        bb_data = self.calculate_bollinger_bands()
        
        fig = go.Figure()
        
        x_data = self.data['Datetime'] if 'Datetime' in self.data.columns else self.data.index
        
        # Upper Band
        fig.add_trace(go.Scatter(
            x=x_data,
            y=bb_data['Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            showlegend=True,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Upper Band: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Lower Band
        fig.add_trace(go.Scatter(
            x=x_data,
            y=bb_data['Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=True,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Lower Band: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Middle Band (SMA)
        fig.add_trace(go.Scatter(
            x=x_data,
            y=bb_data['Middle'],
            mode='lines',
            name='Middle Band (SMA 20)',
            line=dict(color='blue', width=1),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Middle Band: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Price
        fig.add_trace(go.Scatter(
            x=x_data,
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Bollinger Bands',
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
    
    def get_bollinger_position(self) -> str:
        """Get current position relative to Bollinger Bands."""
        bb_data = self.calculate_bollinger_bands()
        current_price = self.data['Close'].iloc[-1]
        current_upper = bb_data['Upper'].iloc[-1]
        current_lower = bb_data['Lower'].iloc[-1]
        current_middle = bb_data['Middle'].iloc[-1]
        
        if current_price > current_upper:
            return "Above Upper Band"
        elif current_price < current_lower:
            return "Below Lower Band"
        elif current_price > current_middle:
            return "Upper Half"
        else:
            return "Lower Half"
    
    def get_trading_signals(self) -> Dict[str, Dict[str, str]]:
        """Generate trading signals based on technical indicators with strength calculation."""
        signals = {}
        
        # RSI Signal
        rsi = self.calculate_rsi()
        current_rsi = rsi.iloc[-1]
        # Calculate RSI strength: distance from overbought/oversold thresholds
        if current_rsi > 70:
            rsi_signal = "SELL"
            rsi_explanation = "RSI above 70 indicates overbought conditions"
            rsi_implication = "Consider taking profits or reducing position size"
            rsi_strength = f"Strong ({(current_rsi - 70):.1f} above overbought)"
        elif current_rsi < 30:
            rsi_signal = "BUY"
            rsi_explanation = "RSI below 30 indicates oversold conditions"
            rsi_implication = "Potential buying opportunity for mean reversion"
            rsi_strength = f"Strong ({(30 - current_rsi):.1f} below oversold)"
        else:
            rsi_signal = "NEUTRAL"
            rsi_explanation = "RSI in neutral zone (30-70)"
            rsi_implication = "No strong momentum signal, monitor for breakouts"
            rsi_distance = min(abs(current_rsi - 70), abs(current_rsi - 30))
            rsi_strength = f"Weak ({rsi_distance:.1f} from threshold)"
        
        signals['RSI'] = {
            'signal': rsi_signal,
            'explanation': rsi_explanation,
            'trading_implication': rsi_implication,
            'strength': rsi_strength
        }
        
        # MACD Signal
        macd_data = self.calculate_macd()
        current_macd = macd_data['MACD'].iloc[-1]
        current_signal = macd_data['Signal'].iloc[-1]
        # Calculate MACD strength: magnitude of MACD-Signal divergence relative to historical volatility
        macd_diff = abs(current_macd - current_signal)
        macd_std = macd_data['MACD'].std() if len(macd_data['MACD']) > 1 else 1
        macd_strength_val = macd_diff / macd_std if macd_std != 0 else 0
        if current_macd > current_signal:
            if len(macd_data['MACD']) > 1 and macd_data['MACD'].iloc[-2] <= macd_data['Signal'].iloc[-2]:
                macd_signal = "BUY"
                macd_explanation = "MACD line crossed above Signal line (bullish crossover)"
                macd_implication = "Potential upward momentum, consider entry"
                macd_strength = f"Strong (Crossover, {macd_diff:.2f} diff)"
            else:
                macd_signal = "BULLISH"
                macd_explanation = "MACD line above Signal line"
                macd_implication = "Positive momentum continues"
                macd_strength = f"Moderate ({macd_strength_val:.2f}x std dev)"
        else:
            if len(macd_data['MACD']) > 1 and macd_data['MACD'].iloc[-2] >= macd_data['Signal'].iloc[-2]:
                macd_signal = "SELL"
                macd_explanation = "MACD line crossed below Signal line (bearish crossover)"
                macd_implication = "Potential downward momentum, consider exit"
                macd_strength = f"Strong (Crossover, {macd_diff:.2f} diff)"
            else:
                macd_signal = "BEARISH"
                macd_explanation = "MACD line below Signal line"
                macd_implication = "Negative momentum continues"
                macd_strength = f"Moderate ({macd_strength_val:.2f}x std dev)"
        
        signals['MACD'] = {
            'signal': macd_signal,
            'explanation': macd_explanation,
            'trading_implication': macd_implication,
            'strength': macd_strength
        }
        
        # Moving Average Signal
        sma_20 = self.calculate_sma(20)
        sma_50 = self.calculate_sma(50)
        current_price = self.data['Close'].iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        # Calculate MA strength: price distance from MAs relative to volatility
        price_volatility = self.data['Close'].pct_change().std() * np.sqrt(252) if len(self.data) > 1 else 0.01
        ma_distance = abs(current_price - current_sma_20) / (current_sma_20 * price_volatility) if current_sma_20 != 0 else 0
        if current_price > current_sma_20 > current_sma_50:
            ma_signal = "BULLISH"
            ma_explanation = "Price above both short and long-term moving averages"
            ma_implication = "Strong upward trend, favorable for long positions"
            ma_strength = f"Strong ({ma_distance:.2f}x volatility)"
        elif current_price < current_sma_20 < current_sma_50:
            ma_signal = "BEARISH"
            ma_explanation = "Price below both short and long-term moving averages"
            ma_implication = "Strong downward trend, consider short positions or exit longs"
            ma_strength = f"Strong ({ma_distance:.2f}x volatility)"
        else:
            ma_signal = "MIXED"
            ma_explanation = "Mixed signals from moving averages"
            ma_implication = "Trend unclear, wait for clearer signals"
            ma_strength = f"Weak ({ma_distance:.2f}x volatility)"
        
        signals['Moving Averages'] = {
            'signal': ma_signal,
            'explanation': ma_explanation,
            'trading_implication': ma_implication,
            'strength': ma_strength
        }
        
        # Bollinger Bands Signal
        bb_data = self.calculate_bollinger_bands()
        bb_position = self.get_bollinger_position()
        # Calculate BB strength: price distance from bands relative to band width
        current_price = self.data['Close'].iloc[-1]
        current_upper = bb_data['Upper'].iloc[-1]
        current_lower = bb_data['Lower'].iloc[-1]
        band_width = (current_upper - current_lower) / current_lower if current_lower != 0 else 0
        bb_distance = abs(current_price - (current_upper if current_price > current_upper else current_lower)) / (band_width * current_lower) if band_width != 0 else 0
        if bb_position == "Above Upper Band":
            bb_signal = "OVERBOUGHT"
            bb_explanation = "Price above upper Bollinger Band"
            bb_implication = "Potential reversal or consolidation expected"
            bb_strength = f"Strong ({bb_distance:.2f}x band width)"
        elif bb_position == "Below Lower Band":
            bb_signal = "OVERSOLD"
            bb_explanation = "Price below lower Bollinger Band"
            bb_implication = "Potential bounce or reversal opportunity"
            bb_strength = f"Strong ({bb_distance:.2f}x band width)"
        else:
            bb_signal = "NORMAL"
            bb_explanation = f"Price in {bb_position.lower()} of Bollinger Bands"
            bb_implication = "Price within normal volatility range"
            bb_strength = f"Weak ({bb_distance:.2f}x band width)"
        
        signals['Bollinger Bands'] = {
            'signal': bb_signal,
            'explanation': bb_explanation,
            'trading_implication': bb_implication,
            'strength': bb_strength
        }
        
        return signals
