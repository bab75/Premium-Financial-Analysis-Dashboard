import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

class PricePredictions:
    """Generate limited price predictions based on technical analysis and statistical methods."""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data.copy()
        self.data = self._prepare_data()
        
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for prediction models."""
        # Handle Datetime column or index
        if 'Datetime' in self.data.columns:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], errors='coerce')
            self.data = self.data.set_index('Datetime')
        elif 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            self.data = self.data.rename(columns={'Date': 'Datetime'})
            self.data = self.data.set_index('Datetime')
        elif pd.api.types.is_datetime64_any_dtype(self.data.index):
            self.data.index = pd.to_datetime(self.data.index, errors='coerce')
        else:
            raise ValueError("No valid Datetime or Date column/index found in data")
        
        # Check for invalid Datetime values
        if self.data.index.isna().any():
            print("Warning: NaT values found in Datetime index")
            self.data = self.data.dropna()
        
        # Sort by date
        self.data = self.data.sort_index()
        
        # Calculate additional features
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['SMA_5'] = self.data['Close'].rolling(window=5, min_periods=1).mean()
        self.data['SMA_10'] = self.data['Close'].rolling(window=10, min_periods=1).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20, min_periods=1).mean()
        self.data['Volatility'] = self.data['Returns'].rolling(window=10, min_periods=1).std()
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=10, min_periods=1).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        
        return self.data
    
    def predict_prices(self, days: int, method: str = "technical_analysis") -> Optional[List[float]]:
        """Generate price predictions using specified method."""
        if len(self.data) < 20:  # Minimum data requirement
            return None
        
        try:
            if method == "technical_analysis":
                return self._predict_technical_analysis(days)
            elif method == "linear_trend":
                return self._predict_linear_trend(days)
            elif method == "moving_average":
                return self._predict_moving_average(days)
            else:
                return self._predict_technical_analysis(days)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def _predict_technical_analysis(self, days: int) -> List[float]:
        """Predict based on technical indicators and momentum."""
        predictions = []
        current_price = self.data['Close'].iloc[-1]
        
        # Calculate technical momentum factors
        sma_5 = self.data['SMA_5'].iloc[-1]
        sma_10 = self.data['SMA_10'].iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        
        # Price momentum relative to moving averages
        momentum_factor = 0
        if current_price > sma_5:
            momentum_factor += 0.3
        if current_price > sma_10:
            momentum_factor += 0.2
        if current_price > sma_20:
            momentum_factor += 0.1
        
        # Trend direction
        if sma_5 > sma_10 > sma_20:
            trend_strength = 1.0  # Strong uptrend
        elif sma_5 < sma_10 < sma_20:
            trend_strength = -1.0  # Strong downtrend
        else:
            trend_strength = 0.0  # Sideways
        
        # Recent volatility
        recent_volatility = self.data['Volatility'].iloc[-5:].mean()
        if pd.isna(recent_volatility):
            recent_volatility = 0.02  # Default 2% volatility
        
        # Volume analysis
        volume_trend = self.data['Volume_Ratio'].iloc[-5:].mean()
        if pd.isna(volume_trend):
            volume_trend = 1.0
        
        volume_factor = min(max(volume_trend - 1.0, -0.1), 0.1)  # Cap at ±10%
        
        # Generate predictions
        for day in range(1, days + 1):
            # Base prediction using momentum and trend
            momentum_change = momentum_factor * trend_strength * 0.01  # 1% max per factor
            
            # Add volume influence
            volume_influence = volume_factor * 0.005  # 0.5% max influence
            
            # Decay factor for longer predictions
            decay_factor = 0.8 ** (day - 1)
            
            # Random walk component with controlled variance
            random_component = np.random.normal(0, recent_volatility * 0.5)
            
            # Combine factors
            daily_change = (momentum_change + volume_influence) * decay_factor + random_component
            
            # Apply change to get predicted price
            if day == 1:
                predicted_price = current_price * (1 + daily_change)
            else:
                predicted_price = predictions[-1] * (1 + daily_change)
            
            predictions.append(predicted_price)
        
        return predictions
    
    def _predict_linear_trend(self, days: int) -> List[float]:
        """Predict using linear regression on recent price trend."""
        # Use last 30 days or all available data if less
        lookback_period = min(30, len(self.data))
        recent_data = self.data.iloc[-lookback_period:].copy()
        
        # Create time features
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['Close'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions
        predictions = []
        for day in range(1, days + 1):
            future_x = len(recent_data) + day - 1
            predicted_price = model.predict([[future_x]])[0]
            
            # Add some realistic variance
            volatility = recent_data['Returns'].std()
            if pd.isna(volatility):
                volatility = 0.02
            
            noise = np.random.normal(0, volatility * predicted_price * 0.3)
            predicted_price += noise
            
            predictions.append(max(predicted_price, 0))  # Ensure non-negative
        
        return predictions
    
    def _predict_moving_average(self, days: int) -> List[float]:
        """Predict using moving average reversion."""
        # Calculate multiple moving averages
        sma_10 = self.data['SMA_10'].iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        current_price = self.data['Close'].iloc[-1]
        
        # Calculate reversion tendency
        reversion_target = (sma_10 + sma_20) / 2
        
        predictions = []
        for day in range(1, days + 1):
            # Mean reversion with decay
            reversion_factor = 0.1 * (0.9 ** (day - 1))  # Stronger reversion initially
            
            if day == 1:
                base_price = current_price
            else:
                base_price = predictions[-1]
            
            # Move towards reversion target
            price_diff = reversion_target - base_price
            predicted_price = base_price + (price_diff * reversion_factor)
            
            # Add volatility
            recent_volatility = self.data['Volatility'].iloc[-10:].mean()
            if pd.isna(recent_volatility):
                recent_volatility = 0.02
            
            noise = np.random.normal(0, recent_volatility * predicted_price * 0.2)
            predicted_price += noise
            
            predictions.append(max(predicted_price, 0))
        
        return predictions
    
    def create_prediction_chart(self, predictions: List[float], days: int) -> go.Figure:
        """Create visualization for price predictions."""
        fig = go.Figure()
        
        # Historical prices (last 30 days)
        lookback_days = min(30, len(self.data))
        historical_data = self.data.iloc[-lookback_days:]
        
        # Historical trace
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Generate future dates
        last_date = historical_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>' +
                          'Predicted: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Connect last historical point to first prediction
        fig.add_trace(go.Scatter(
            x=[historical_data.index[-1], future_dates[0]],
            y=[historical_data['Close'].iloc[-1], predictions[0]],
            mode='lines',
            name='Connection',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>' +
                          'Price: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        # Add confidence bands (simple approximation)
        recent_volatility = self.data['Volatility'].iloc[-10:].mean()
        if pd.isna(recent_volatility):
            recent_volatility = 0.02
        
        upper_band = [p * (1 + recent_volatility * 2) for p in predictions]
        lower_band = [p * (1 - recent_volatility * 2) for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_band,
            mode='lines',
            name='Upper Confidence',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            showlegend=False,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>' +
                          'Upper: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_band,
            mode='lines',
            name='Lower Confidence',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%Y}<br>' +
                          'Lower: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Price Predictions - Next {days} Days',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(
                type='date',
                showticklabels=True,  # Show date labels on axis
                showgrid=True,
                hoverformat='%m-%d-%Y'  # MM-DD-YYYY format on hover
            )
        )
        
        return fig
    
    def calculate_prediction_confidence(self) -> Dict[str, any]:
        """Calculate confidence metrics for predictions."""
        # Historical accuracy simulation
        accuracy_scores = []
        
        # Test prediction accuracy on historical data
        if len(self.data) >= 50:  # Need sufficient data for backtesting
            for i in range(5):  # Test 5 different periods
                test_start = -(20 + i * 5)  # Different starting points
                test_end = -(10 + i * 5)
                
                if abs(test_start) < len(self.data):
                    # Create subset for testing
                    test_data = self.data.iloc[:test_start].copy()
                    actual_prices = self.data.iloc[test_start:test_end]['Close'].values
                    
                    # Generate predictions using the test data
                    temp_predictor = PricePredictions(test_data)
                    test_predictions = temp_predictor._predict_technical_analysis(len(actual_prices))
                    
                    if test_predictions:
                        mae = mean_absolute_error(actual_prices, test_predictions)
                        accuracy = 1 - (mae / np.mean(actual_prices))
                        accuracy_scores.append(max(0, accuracy))
        
        # Calculate confidence metrics
        if accuracy_scores:
            avg_accuracy = np.mean(accuracy_scores)
            confidence_level = min(max(avg_accuracy * 100, 10), 85)  # Cap between 10-85%
        else:
            confidence_level = 50  # Default moderate confidence
        
        # Volatility risk
        recent_volatility = self.data['Volatility'].iloc[-10:].mean()
        if pd.isna(recent_volatility):
            recent_volatility = 0.02
        
        volatility_risk = recent_volatility * 100
        
        # Trend strength
        sma_5 = self.data['SMA_5'].iloc[-1]
        sma_10 = self.data['SMA_10'].iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        current_price = self.data['Close'].iloc[-1]
        
        if current_price > sma_5 > sma_10 > sma_20:
            trend_strength = "Strong Uptrend"
        elif current_price < sma_5 < sma_10 < sma_20:
            trend_strength = "Strong Downtrend"
        elif current_price > sma_20:
            trend_strength = "Moderate Uptrend"
        elif current_price < sma_20:
            trend_strength = "Moderate Downtrend"
        else:
            trend_strength = "Sideways"
        
        return {
            'confidence_level': f"{confidence_level:.0f}%",
            'volatility_risk': volatility_risk,
            'trend_strength': trend_strength,
            'data_quality': "Good" if len(self.data) >= 100 else "Limited",
            'prediction_reliability': "Educational Only"
        }
    
    def get_prediction_disclaimer(self) -> str:
        """Return disclaimer for predictions."""
        return """
        ⚠️ IMPORTANT DISCLAIMER:
        
        These price predictions are for educational and analytical purposes only.
        They are based on limited technical analysis and statistical methods.
        
        • Not investment advice
        • Past performance does not guarantee future results
        • Market conditions can change rapidly
        • Consider multiple factors before making investment decisions
        • Consult with qualified financial advisors
        
        Use these predictions as one of many tools in your analysis toolkit.
        """
