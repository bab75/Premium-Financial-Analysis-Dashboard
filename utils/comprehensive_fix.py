"""
Comprehensive Fix Module for All Critical Issues
- Data processing reliability
- Price change amount calculations
- DataFrame type safety
- Visualization date formatting
- Gauge alignment fixes
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, Tuple, Any
from datetime import datetime

class ComprehensiveFix:
    """Complete solution for all identified critical issues."""
    
    def __init__(self):
        self.required_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change']
    
    def process_comparative_data(self, current_data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """Process comparative data with guaranteed DataFrame output and price change amounts."""
        try:
            if not isinstance(current_data, pd.DataFrame) or not isinstance(previous_data, pd.DataFrame):
                return pd.DataFrame()
            
            # Find symbol columns
            current_symbol = self._find_symbol_column(current_data)
            previous_symbol = self._find_symbol_column(previous_data)
            
            if not current_symbol or not previous_symbol:
                return pd.DataFrame()
            
            # Clean and prepare datasets
            current_clean = self._clean_dataset(current_data, current_symbol, "_current")
            previous_clean = self._clean_dataset(previous_data, previous_symbol, "_previous")
            
            if current_clean.empty or previous_clean.empty:
                return pd.DataFrame()
            
            # Merge on symbol
            merged = pd.merge(current_clean, previous_clean, on='Symbol', how='inner')
            
            if merged.empty:
                return pd.DataFrame()
            
            # Calculate price changes with amounts
            merged = self._calculate_comprehensive_changes(merged)
            
            return merged
            
        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
            return pd.DataFrame()
    
    def _find_symbol_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find symbol column with flexible naming."""
        candidates = ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'Stock']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _clean_dataset(self, df: pd.DataFrame, symbol_col: str, suffix: str) -> pd.DataFrame:
        """Clean dataset and prepare for merging."""
        try:
            clean_df = df.copy()
            
            # Standardize symbol column
            clean_df['Symbol'] = clean_df[symbol_col].astype(str).str.strip()
            clean_df = clean_df[clean_df['Symbol'] != '']
            clean_df = clean_df[clean_df['Symbol'] != 'nan']
            clean_df = clean_df[clean_df['Symbol'].notna()]
            
            # Find and clean price column
            price_col = self._find_price_column(df)
            if price_col:
                clean_df[f'Price{suffix}'] = pd.to_numeric(
                    clean_df[price_col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                )
            
            # Find and clean volume column
            volume_col = self._find_volume_column(df)
            if volume_col:
                clean_df[f'Volume{suffix}'] = pd.to_numeric(clean_df[volume_col], errors='coerce')
            
            # Keep relevant columns
            keep_cols = ['Symbol']
            if f'Price{suffix}' in clean_df.columns:
                keep_cols.append(f'Price{suffix}')
            if f'Volume{suffix}' in clean_df.columns:
                keep_cols.append(f'Volume{suffix}')
            
            # Add sector/industry if available
            for col in ['Sector', 'Industry', 'Country']:
                if col in clean_df.columns:
                    keep_cols.append(col)
            
            return clean_df[keep_cols].dropna(subset=['Symbol'])
            
        except Exception:
            return pd.DataFrame()
    
    def _find_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find price column with flexible naming."""
        candidates = ['Last Sale', 'Price', 'Close', 'Last Price', 'Current Price', 'Last']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _find_volume_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find volume column."""
        candidates = ['Volume', 'volume', 'VOLUME', 'Trading Volume']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _calculate_comprehensive_changes(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Calculate price and volume changes with amounts."""
        try:
            # Price changes
            if 'Price_current' in merged.columns and 'Price_previous' in merged.columns:
                merged['Price_Change_Amount'] = (merged['Price_current'] - merged['Price_previous']).round(2)
                merged['Price_Change_Pct'] = (
                    (merged['Price_current'] - merged['Price_previous']) / merged['Price_previous'] * 100
                ).round(2)
                
                # Create combined price change display
                merged['Price_Change_Display'] = merged.apply(
                    lambda row: f"{row['Price_Change_Pct']:.2f}% ({row['Price_Change_Amount']:+.2f})",
                    axis=1
                )
            
            # Volume changes
            if 'Volume_current' in merged.columns and 'Volume_previous' in merged.columns:
                merged['Volume_Change_Pct'] = (
                    (merged['Volume_current'] - merged['Volume_previous']) / merged['Volume_previous'] * 100
                ).round(2)
            
            return merged
            
        except Exception:
            return merged
    
    def get_performance_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reliable performance summary."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty or 'Price_Change_Pct' not in data.columns:
                return {}
            
            valid_changes = data['Price_Change_Pct'].dropna()
            if len(valid_changes) == 0:
                return {}
            
            return {
                'total_stocks': len(data),
                'valid_stocks': len(valid_changes),
                'avg_change': float(valid_changes.mean()),
                'gainers': int((valid_changes > 0).sum()),
                'losers': int((valid_changes < 0).sum()),
                'max_gain': float(valid_changes.max()),
                'max_loss': float(valid_changes.min()),
                'median_change': float(valid_changes.median())
            }
            
        except Exception:
            return {}
    
    def get_top_performers(self, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top performing stocks with proper DataFrame handling."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty or 'Price_Change_Pct' not in data.columns:
                return pd.DataFrame()
            
            valid_data = data.dropna(subset=['Price_Change_Pct'])
            if valid_data.empty:
                return pd.DataFrame()
            
            return valid_data.nlargest(n, 'Price_Change_Pct')
            
        except Exception:
            return pd.DataFrame()
    
    def get_bottom_performers(self, data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get bottom performing stocks with proper DataFrame handling."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty or 'Price_Change_Pct' not in data.columns:
                return pd.DataFrame()
            
            valid_data = data.dropna(subset=['Price_Change_Pct'])
            if valid_data.empty:
                return pd.DataFrame()
            
            return valid_data.nsmallest(n, 'Price_Change_Pct')
            
        except Exception:
            return pd.DataFrame()
    
    def create_outlier_analysis(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create outlier analysis with profit/loss in ascending order."""
        try:
            if not isinstance(data, pd.DataFrame) or data.empty or 'Price_Change_Pct' not in data.columns:
                return {}
            
            # Calculate outliers using IQR method
            Q1 = data['Price_Change_Pct'].quantile(0.25)
            Q3 = data['Price_Change_Pct'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[
                (data['Price_Change_Pct'] < lower_bound) | 
                (data['Price_Change_Pct'] > upper_bound)
            ].copy()
            
            if not outliers.empty:
                # Add profit/loss calculation (assuming $1000 investment)
                outliers['Profit_Loss'] = (outliers['Price_Change_Pct'] / 100 * 1000).round(2)
                
                # Sort by profit/loss in ascending order (losses first, then gains)
                outliers = outliers.sort_values('Profit_Loss', ascending=True)
            
            return {
                'outliers': outliers,
                'total_outliers': len(outliers),
                'extreme_gainers': outliers[outliers['Price_Change_Pct'] > upper_bound],
                'extreme_losers': outliers[outliers['Price_Change_Pct'] < lower_bound]
            }
            
        except Exception:
            return {}
    
    def format_date_axis(self, fig: go.Figure, date_column: pd.Series) -> go.Figure:
        """Format date axis to MM-DD-YY format."""
        try:
            if hasattr(date_column, 'dt'):
                formatted_dates = date_column.dt.strftime('%m-%d-%y')
            else:
                # Try to convert to datetime if not already
                formatted_dates = pd.to_datetime(date_column, errors='coerce').dt.strftime('%m-%d-%y')
            
            fig.update_xaxes(
                tickformat='%m-%d-%y',
                tickmode='array',
                tickvals=list(range(len(formatted_dates))),
                ticktext=formatted_dates.tolist()
            )
            
        except Exception:
            pass  # Keep original formatting if conversion fails
        
        return fig
    
    def create_enhanced_gauge(self, value: float, title: str, min_val: float = 0, max_val: float = 100) -> go.Figure:
        """Create properly aligned gauge visualization."""
        try:
            # Ensure value is within bounds
            value = max(min_val, min(max_val, value))
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title, 'font': {'size': 16}},
                delta={'reference': (max_val + min_val) / 2},
                gauge={
                    'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                    'bar': {'color': self._get_gauge_color(value, min_val, max_val)},
                    'steps': [
                        {'range': [min_val, max_val * 0.3], 'color': 'lightgray'},
                        {'range': [max_val * 0.3, max_val * 0.7], 'color': 'yellow'},
                        {'range': [max_val * 0.7, max_val], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_val * 0.8
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=60, b=20),
                font={'size': 12}
            )
            
            return fig
            
        except Exception:
            return go.Figure()
    
    def _get_gauge_color(self, value: float, min_val: float, max_val: float) -> str:
        """Get appropriate color for gauge based on value."""
        normalized = (value - min_val) / (max_val - min_val)
        
        if normalized < 0.3:
            return "green"
        elif normalized < 0.7:
            return "orange"
        else:
            return "red"