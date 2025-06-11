import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class Analytics:
    """Generate comprehensive analytics and insights from financial data."""
    
    def __init__(self, daily_data: Optional[pd.DataFrame] = None, historical_data: Optional[pd.DataFrame] = None):
        self.daily_data = daily_data
        self.historical_data = historical_data
    
    def get_top_performers(self, n: int = 5) -> pd.DataFrame:
        """Get top N performing stocks by % change."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        top_performers = self.daily_data.nlargest(n, '% Change')[
            ['Symbol', 'Name', 'Last Sale', '% Change', 'Volume', 'Sector']
        ].round(2)
        
        return top_performers
    
    def get_bottom_performers(self, n: int = 5) -> pd.DataFrame:
        """Get bottom N performing stocks by % change."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        bottom_performers = self.daily_data.nsmallest(n, '% Change')[
            ['Symbol', 'Name', 'Last Sale', '% Change', 'Volume', 'Sector']
        ].round(2)
        
        return bottom_performers
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Analyze performance by sector."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        sector_analysis = self.daily_data.groupby('Sector').agg({
            '% Change': ['mean', 'std', 'count'],
            'Volume': 'mean',
            'Market Cap': 'mean',
            'Last Sale': 'mean'
        }).round(2)
        
        # Flatten column names
        sector_analysis.columns = [
            'Avg_Change_Pct', 'Volatility', 'Stock_Count',
            'Avg_Volume', 'Avg_Market_Cap', 'Avg_Price'
        ]
        
        # Sort by average change
        sector_analysis = sector_analysis.sort_values('Avg_Change_Pct', ascending=False)
        
        return sector_analysis.reset_index()
    
    def get_industry_analysis(self) -> pd.DataFrame:
        """Analyze performance by industry (top 20)."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        # Get top industries by stock count
        top_industries = self.daily_data['Industry'].value_counts().head(20).index
        
        industry_data = self.daily_data[self.daily_data['Industry'].isin(top_industries)]
        
        industry_analysis = industry_data.groupby('Industry').agg({
            '% Change': ['mean', 'std', 'count'],
            'Volume': 'mean',
            'Market Cap': 'mean'
        }).round(2)
        
        industry_analysis.columns = [
            'Avg_Change_Pct', 'Volatility', 'Stock_Count',
            'Avg_Volume', 'Avg_Market_Cap'
        ]
        
        return industry_analysis.sort_values('Avg_Change_Pct', ascending=False).reset_index()
    
    def get_country_distribution(self) -> pd.DataFrame:
        """Get distribution of stocks by country."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        country_dist = self.daily_data['Country'].value_counts().head(15)
        country_pct = (country_dist / len(self.daily_data) * 100).round(1)
        
        result = pd.DataFrame({
            'Country': country_dist.index,
            'Stock_Count': country_dist.values,
            'Percentage': country_pct.values
        })
        
        return result
    
    def get_market_cap_distribution(self) -> pd.DataFrame:
        """Analyze market cap distribution."""
        if self.daily_data is None:
            return pd.DataFrame()
        
        # Define market cap categories
        conditions = [
            self.daily_data['Market Cap'] < 1e9,
            (self.daily_data['Market Cap'] >= 1e9) & (self.daily_data['Market Cap'] < 10e9),
            (self.daily_data['Market Cap'] >= 10e9) & (self.daily_data['Market Cap'] < 50e9),
            self.daily_data['Market Cap'] >= 50e9
        ]
        
        categories = ['Small Cap (<$1B)', 'Mid Cap ($1B-$10B)', 'Large Cap ($10B-$50B)', 'Mega Cap (>$50B)']
        
        self.daily_data['Market_Cap_Category'] = np.select(conditions, categories, default='Unknown')
        
        cap_distribution = self.daily_data.groupby('Market_Cap_Category').agg({
            '% Change': 'mean',
            'Volume': 'mean',
            'Symbol': 'count'
        }).round(2)
        
        cap_distribution.columns = ['Avg_Change_Pct', 'Avg_Volume', 'Stock_Count']
        
        return cap_distribution.reset_index()
    
    def get_historical_summary(self) -> Dict:
        """Generate summary statistics for historical data."""
        if self.historical_data is None:
            return {}
        
        # Calculate returns
        returns = self.historical_data['Close'].pct_change().dropna()
        
        # Price statistics
        price_stats = {
            'avg_daily_range': (self.historical_data['High'] - self.historical_data['Low']).mean(),
            'volatility': returns.std() * 100,  # As percentage
            'total_return': ((self.historical_data['Close'].iloc[-1] / self.historical_data['Close'].iloc[0]) - 1) * 100,
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_volume': self.historical_data['Volume'].mean(),
            'price_range_low': self.historical_data['Low'].min(),
            'price_range_high': self.historical_data['High'].max(),
            'trading_days': len(self.historical_data)
        }
        
        return price_stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.historical_data is None:
            return 0.0
        
        prices = self.historical_data['Close']
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min() * 100  # As percentage
    
    def get_comparative_analysis(self, symbol: str) -> Dict:
        """Compare individual stock with sector and industry averages."""
        if self.daily_data is None or self.historical_data is None:
            return {}
        
        # Find stock in daily data
        stock_data = self.daily_data[self.daily_data['Symbol'] == symbol]
        
        if stock_data.empty:
            return {'error': f'Symbol {symbol} not found in daily data'}
        
        stock_info = stock_data.iloc[0]
        sector = stock_info['Sector']
        industry = stock_info['Industry']
        
        # Sector comparison
        sector_stocks = self.daily_data[self.daily_data['Sector'] == sector]
        sector_avg = {
            'avg_change': sector_stocks['% Change'].mean(),
            'avg_volume': sector_stocks['Volume'].mean(),
            'avg_market_cap': sector_stocks['Market Cap'].mean(),
            'stock_count': len(sector_stocks)
        }
        
        # Industry comparison
        industry_stocks = self.daily_data[self.daily_data['Industry'] == industry]
        industry_avg = {
            'avg_change': industry_stocks['% Change'].mean(),
            'avg_volume': industry_stocks['Volume'].mean(),
            'avg_market_cap': industry_stocks['Market Cap'].mean(),
            'stock_count': len(industry_stocks)
        }
        
        # Historical performance comparison
        historical_metrics = self.get_historical_summary()
        
        comparison = {
            'stock_info': {
                'symbol': symbol,
                'name': stock_info['Name'],
                'sector': sector,
                'industry': industry,
                'current_change': stock_info['% Change'],
                'current_volume': stock_info['Volume'],
                'market_cap': stock_info['Market Cap']
            },
            'sector_comparison': {
                'sector': sector,
                'stock_vs_sector_change': stock_info['% Change'] - sector_avg['avg_change'],
                'stock_vs_sector_volume': stock_info['Volume'] / sector_avg['avg_volume'] if sector_avg['avg_volume'] > 0 else 1,
                'sector_rank': int((sector_stocks['% Change'] > stock_info['% Change']).sum() + 1),
                'sector_total': len(sector_stocks),
                **sector_avg
            },
            'industry_comparison': {
                'industry': industry,
                'stock_vs_industry_change': stock_info['% Change'] - industry_avg['avg_change'],
                'stock_vs_industry_volume': stock_info['Volume'] / industry_avg['avg_volume'] if industry_avg['avg_volume'] > 0 else 1,
                'industry_rank': int((industry_stocks['% Change'] > stock_info['% Change']).sum() + 1),
                'industry_total': len(industry_stocks),
                **industry_avg
            },
            'historical_performance': historical_metrics
        }
        
        return comparison
    
    def analyze_patterns(self) -> Dict:
        """Analyze seasonal and volume patterns."""
        patterns = {
            'seasonal_patterns': [],
            'volume_patterns': []
        }
        
        if self.historical_data is None:
            return patterns
        
        # Seasonal patterns (if we have date index)
        if pd.api.types.is_datetime64_any_dtype(self.historical_data.index):
            try:
                # Day of week patterns
                self.historical_data['DayOfWeek'] = self.historical_data.index.dayofweek
                self.historical_data['Returns'] = self.historical_data['Close'].pct_change()
                
                dow_returns = self.historical_data.groupby('DayOfWeek')['Returns'].mean()
                best_day = dow_returns.idxmax()
                worst_day = dow_returns.idxmin()
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                if abs(dow_returns.iloc[best_day]) > 0.001:  # Threshold for significance
                    patterns['seasonal_patterns'].append(
                        f"Best performing day: {day_names[best_day]} (+{dow_returns.iloc[best_day]*100:.2f}% avg)"
                    )
                
                if abs(dow_returns.iloc[worst_day]) > 0.001:
                    patterns['seasonal_patterns'].append(
                        f"Worst performing day: {day_names[worst_day]} ({dow_returns.iloc[worst_day]*100:.2f}% avg)"
                    )
                
                # Month patterns
                if len(self.historical_data) > 60:  # At least 2 months of data
                    self.historical_data['Month'] = self.historical_data.index.month
                    month_returns = self.historical_data.groupby('Month')['Returns'].mean()
                    
                    best_month = month_returns.idxmax()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    if abs(month_returns.iloc[best_month-1]) > 0.01:
                        patterns['seasonal_patterns'].append(
                            f"Best performing month: {month_names[best_month-1]} (+{month_returns.iloc[best_month-1]*100:.2f}% avg)"
                        )
            except Exception:
                pass
        
        # Volume patterns
        try:
            volume_ma = self.historical_data['Volume'].rolling(window=20, min_periods=1).mean()
            high_volume_days = self.historical_data['Volume'] > (volume_ma * 1.5)
            
            if high_volume_days.sum() > 5:  # Need some high volume days
                high_vol_returns = self.historical_data.loc[high_volume_days, 'Returns'].mean() if 'Returns' in self.historical_data.columns else 0
                normal_vol_returns = self.historical_data.loc[~high_volume_days, 'Returns'].mean() if 'Returns' in self.historical_data.columns else 0
                
                if abs(high_vol_returns - normal_vol_returns) > 0.005:
                    if high_vol_returns > normal_vol_returns:
                        patterns['volume_patterns'].append(
                            f"High volume days tend to be positive (+{high_vol_returns*100:.2f}% vs {normal_vol_returns*100:.2f}%)"
                        )
                    else:
                        patterns['volume_patterns'].append(
                            f"High volume days tend to be negative ({high_vol_returns*100:.2f}% vs {normal_vol_returns*100:.2f}%)"
                        )
            
            # Volume trend analysis
            recent_volume = self.historical_data['Volume'].iloc[-10:].mean()
            historical_volume = self.historical_data['Volume'].iloc[:-10].mean()
            
            if len(self.historical_data) > 20:
                volume_change = (recent_volume - historical_volume) / historical_volume
                
                if abs(volume_change) > 0.2:  # 20% change threshold
                    if volume_change > 0:
                        patterns['volume_patterns'].append(
                            f"Recent volume increase: +{volume_change*100:.1f}% above historical average"
                        )
                    else:
                        patterns['volume_patterns'].append(
                            f"Recent volume decrease: {volume_change*100:.1f}% below historical average"
                        )
        except Exception:
            pass
        
        return patterns
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics."""
        risk_metrics = {}
        
        if self.historical_data is None:
            return risk_metrics
        
        try:
            # Calculate returns
            returns = self.historical_data['Close'].pct_change().dropna()
            
            if len(returns) > 20:
                # Basic risk metrics
                volatility = returns.std() * np.sqrt(252)  # Annualized
                risk_metrics['annualized_volatility'] = f"{volatility * 100:.1f}%"
                
                # Sharpe ratio (assuming 3% risk-free rate)
                risk_free_rate = 0.03
                excess_returns = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                risk_metrics['sharpe_ratio'] = f"{sharpe_ratio:.2f}"
                
                # Maximum drawdown (already calculated)
                max_dd = self._calculate_max_drawdown()
                risk_metrics['max_drawdown'] = f"{max_dd:.1f}"
                
                # Value at Risk (95% confidence)
                var_95 = np.percentile(returns, 5) * 100
                risk_metrics['var_95'] = f"{var_95:.1f}"
                
                # Beta calculation (simplified, using market proxy)
                # Since we don't have market data, we'll use sector comparison if available
                if self.daily_data is not None:
                    try:
                        stock_symbol = self._find_stock_symbol()
                        if stock_symbol:
                            stock_info = self.daily_data[self.daily_data['Symbol'] == stock_symbol]
                            if not stock_info.empty:
                                sector = stock_info.iloc[0]['Sector']
                                sector_avg_change = self.daily_data[
                                    self.daily_data['Sector'] == sector
                                ]['% Change'].mean()
                                
                                # Simplified beta calculation
                                stock_change = stock_info.iloc[0]['% Change']
                                beta_proxy = stock_change / sector_avg_change if sector_avg_change != 0 else 1.0
                                risk_metrics['beta'] = f"{beta_proxy:.2f}"
                    except Exception:
                        risk_metrics['beta'] = "N/A"
                
        except Exception as e:
            risk_metrics['error'] = f"Error calculating risk metrics: {str(e)}"
        
        return risk_metrics
    
    def _find_stock_symbol(self) -> Optional[str]:
        """Try to find the stock symbol that matches historical data."""
        if self.daily_data is None or self.historical_data is None:
            return None
        
        # Simple heuristic: find symbol with closest volume or price
        try:
            hist_avg_volume = self.historical_data['Volume'].mean()
            hist_avg_price = self.historical_data['Close'].mean()
            
            # Find best match based on volume and price similarity
            volume_diff = abs(self.daily_data['Volume'] - hist_avg_volume)
            price_diff = abs(self.daily_data['Last Sale'] - hist_avg_price)
            
            # Normalize differences
            volume_diff_norm = volume_diff / self.daily_data['Volume'].std()
            price_diff_norm = price_diff / self.daily_data['Last Sale'].std()
            
            combined_diff = volume_diff_norm + price_diff_norm
            best_match_idx = combined_diff.idxmin()
            
            return self.daily_data.loc[best_match_idx, 'Symbol']
        except Exception:
            return None
    
    def generate_trading_strategies(self, trading_signals: Dict) -> List[Dict]:
        """Generate trading strategy recommendations based on signals."""
        strategies = []
        
        # Momentum Strategy
        momentum_signals = []
        if 'RSI' in trading_signals:
            momentum_signals.append(trading_signals['RSI']['signal'])
        if 'MACD' in trading_signals:
            momentum_signals.append(trading_signals['MACD']['signal'])
        
        bullish_signals = sum(1 for signal in momentum_signals if signal in ['BUY', 'BULLISH'])
        bearish_signals = sum(1 for signal in momentum_signals if signal in ['SELL', 'BEARISH'])
        
        if bullish_signals > bearish_signals:
            strategies.append({
                'name': 'Momentum Trading',
                'type': 'Bullish Momentum',
                'risk_level': 'Medium-High',
                'time_horizon': '1-4 weeks',
                'description': 'Take advantage of positive momentum indicators',
                'entry_conditions': 'RSI above 50, MACD bullish crossover, price above moving averages',
                'exit_conditions': 'RSI overbought (>70), MACD bearish crossover, or 10% profit target',
                'risk_management': 'Stop loss at 5-8% below entry, position size 2-3% of portfolio'
            })
        elif bearish_signals > bullish_signals:
            strategies.append({
                'name': 'Contrarian Entry',
                'type': 'Counter-trend',
                'risk_level': 'High',
                'time_horizon': '1-2 weeks',
                'description': 'Wait for oversold conditions to reverse',
                'entry_conditions': 'RSI below 30, significant price decline, high volume',
                'exit_conditions': 'RSI above 50, price recovers to moving average',
                'risk_management': 'Tight stop loss at 3-5%, small position size 1-2% of portfolio'
            })
        
        # Mean Reversion Strategy
        if 'Bollinger Bands' in trading_signals:
            bb_signal = trading_signals['Bollinger Bands']['signal']
            
            if bb_signal == 'OVERSOLD':
                strategies.append({
                    'name': 'Mean Reversion',
                    'type': 'Reversal',
                    'risk_level': 'Medium',
                    'time_horizon': '1-3 weeks',
                    'description': 'Buy when price is below lower Bollinger Band',
                    'entry_conditions': 'Price below lower BB, RSI oversold, volume confirmation',
                    'exit_conditions': 'Price returns to middle BB or upper BB',
                    'risk_management': 'Stop loss below recent swing low, 2-4% position size'
                })
            elif bb_signal == 'OVERBOUGHT':
                strategies.append({
                    'name': 'Overbought Fade',
                    'type': 'Short-term Reversal',
                    'risk_level': 'Medium-High',
                    'time_horizon': '3-10 days',
                    'description': 'Short-term pullback expected from overbought levels',
                    'entry_conditions': 'Price above upper BB, RSI overbought, decreasing volume',
                    'exit_conditions': 'Price returns to middle BB, RSI normalizes',
                    'risk_management': 'Quick exit if momentum continues, 1-2% position size'
                })
        
        # Trend Following Strategy
        if 'Moving Averages' in trading_signals:
            ma_signal = trading_signals['Moving Averages']['signal']
            
            if ma_signal == 'BULLISH':
                strategies.append({
                    'name': 'Trend Following',
                    'type': 'Long-term Bullish',
                    'risk_level': 'Medium',
                    'time_horizon': '2-8 weeks',
                    'description': 'Follow established uptrend with moving average support',
                    'entry_conditions': 'Price above all major MAs, MAs in bullish alignment',
                    'exit_conditions': 'Price breaks below key MA, trend reversal signals',
                    'risk_management': 'Trailing stop below key MA, 3-5% position size'
                })
            elif ma_signal == 'BEARISH':
                strategies.append({
                    'name': 'Trend Avoidance',
                    'type': 'Defensive',
                    'risk_level': 'Low',
                    'time_horizon': 'Until trend changes',
                    'description': 'Avoid long positions during downtrend',
                    'entry_conditions': 'Wait for trend reversal signals',
                    'exit_conditions': 'N/A - staying out of position',
                    'risk_management': 'Preserve capital, consider defensive sectors'
                })
        
        # Conservative Strategy (always include)
        strategies.append({
            'name': 'Conservative Approach',
            'type': 'Low Risk',
            'risk_level': 'Low',
            'time_horizon': 'Long-term',
            'description': 'Wait for clearer signals and focus on risk management',
            'entry_conditions': 'Multiple confirming signals, strong fundamental support',
            'exit_conditions': 'Predetermined profit targets or stop losses',
            'risk_management': 'Small position sizes, diversification, dollar-cost averaging'
        })
        
        return strategies
