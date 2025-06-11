"""
Enhanced Data Processor with Robust Error Handling
Fixes all data type issues and ensures proper DataFrame operations
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    """Enhanced data processor with comprehensive error handling and type safety."""
    
    def __init__(self):
        self.current_data = None
        self.previous_data = None
        self.merged_data = None
        
    def process_comparative_data(self, current_data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """Process and merge datasets for comparative analysis."""
        try:
            # Ensure inputs are DataFrames
            if not isinstance(current_data, pd.DataFrame) or not isinstance(previous_data, pd.DataFrame):
                st.error("Input data must be pandas DataFrames")
                return pd.DataFrame()
            
            if current_data.empty or previous_data.empty:
                st.warning("One or both datasets are empty")
                return pd.DataFrame()
            
            # Find symbol columns
            current_symbol_col = self._find_symbol_column(current_data)
            previous_symbol_col = self._find_symbol_column(previous_data)
            
            if not current_symbol_col or not previous_symbol_col:
                st.error("Symbol column not found in datasets")
                return pd.DataFrame()
            
            # Prepare datasets for merging
            current_clean = self._prepare_dataset(current_data.copy(), current_symbol_col, '_current')
            previous_clean = self._prepare_dataset(previous_data.copy(), previous_symbol_col, '_previous')
            
            # Merge on Symbol
            merged = pd.merge(
                current_clean,
                previous_clean,
                on='Symbol',
                how='inner',
                suffixes=('_current', '_previous')
            )
            
            if merged.empty:
                st.warning(f"No matching symbols found. Current: {len(current_clean)}, Previous: {len(previous_clean)}")
                return pd.DataFrame()
            
            # Process price data and calculate changes
            merged = self._process_price_changes(merged)
            
            # Process volume data
            merged = self._process_volume_changes(merged)
            
            # Add final validations
            merged = merged.dropna(subset=['Symbol'])
            merged = merged[merged['Symbol'].str.strip() != '']
            
            self.merged_data = merged
            return merged
            
        except Exception as e:
            st.error(f"Error processing comparative data: {str(e)}")
            return pd.DataFrame()
    
    def _find_symbol_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find symbol column in dataset."""
        symbol_candidates = ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock']
        for col in df.columns:
            if col in symbol_candidates:
                return col
            if 'symbol' in col.lower() or 'ticker' in col.lower():
                return col
        return None
    
    def _prepare_dataset(self, df: pd.DataFrame, symbol_col: str, suffix: str) -> pd.DataFrame:
        """Prepare dataset for merging."""
        # Rename symbol column to standard name
        if symbol_col != 'Symbol':
            df = df.rename(columns={symbol_col: 'Symbol'})
        
        # Clean symbols
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
        
        # Remove invalid symbols
        df = df[df['Symbol'].notna() & (df['Symbol'] != '') & (df['Symbol'] != 'NAN')]
        
        return df
    
    def _process_price_changes(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Process price changes in merged data."""
        try:
            # Look for existing percentage change columns first
            pct_change_cols = [col for col in merged.columns if '% Change' in col and '_current' in col]
            
            if pct_change_cols:
                # Use existing percentage change data
                pct_col = pct_change_cols[0]
                merged['Price_Change_Pct'] = pd.to_numeric(
                    merged[pct_col].astype(str).str.replace('%', '').str.replace(',', ''),
                    errors='coerce'
                )
                
                # Look for net change column
                net_change_cols = [col for col in merged.columns if 'Net Change' in col and '_current' in col]
                if net_change_cols:
                    net_col = net_change_cols[0]
                    merged['Price_Change'] = pd.to_numeric(
                        merged[net_col].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
            else:
                # Calculate from price columns
                current_price_col = self._find_price_column(merged, '_current')
                previous_price_col = self._find_price_column(merged, '_previous')
                
                if current_price_col and previous_price_col:
                    current_prices = pd.to_numeric(
                        merged[current_price_col].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
                    previous_prices = pd.to_numeric(
                        merged[previous_price_col].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
                    
                    merged['Last Sale_current'] = current_prices
                    merged['Last Sale_previous'] = previous_prices
                    merged['Price_Change'] = current_prices - previous_prices
                    
                    # Calculate percentage change safely
                    valid_mask = (previous_prices > 0) & (previous_prices.notna())
                    merged['Price_Change_Pct'] = 0.0
                    merged.loc[valid_mask, 'Price_Change_Pct'] = (
                        (current_prices[valid_mask] / previous_prices[valid_mask] - 1) * 100
                    )
            
            # Ensure Price_Change_Pct exists and is numeric
            if 'Price_Change_Pct' not in merged.columns:
                merged['Price_Change_Pct'] = 0.0
            else:
                merged['Price_Change_Pct'] = pd.to_numeric(merged['Price_Change_Pct'], errors='coerce').fillna(0)
            
            return merged
            
        except Exception as e:
            st.warning(f"Error processing price changes: {str(e)}")
            merged['Price_Change_Pct'] = 0.0
            return merged
    
    def _process_volume_changes(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Process volume changes in merged data."""
        try:
            current_volume_col = self._find_volume_column(merged, '_current')
            previous_volume_col = self._find_volume_column(merged, '_previous')
            
            if current_volume_col and previous_volume_col:
                current_volumes = pd.to_numeric(
                    merged[current_volume_col].astype(str).str.replace(',', ''),
                    errors='coerce'
                )
                previous_volumes = pd.to_numeric(
                    merged[previous_volume_col].astype(str).str.replace(',', ''),
                    errors='coerce'
                )
                
                merged['Volume_current'] = current_volumes
                merged['Volume_previous'] = previous_volumes
                merged['Volume_Change'] = current_volumes - previous_volumes
                
                # Calculate volume change percentage safely
                valid_mask = (previous_volumes > 0) & (previous_volumes.notna())
                merged['Volume_Change_Pct'] = 0.0
                merged.loc[valid_mask, 'Volume_Change_Pct'] = (
                    (current_volumes[valid_mask] / previous_volumes[valid_mask] - 1) * 100
                )
            
            return merged
            
        except Exception as e:
            st.warning(f"Error processing volume changes: {str(e)}")
            return merged
    
    def _find_price_column(self, df: pd.DataFrame, suffix: str) -> Optional[str]:
        """Find price column with given suffix."""
        price_candidates = ['Last Sale', 'Price', 'Close', 'Last', 'Current Price']
        for col in df.columns:
            if suffix in col:
                for candidate in price_candidates:
                    if candidate in col:
                        return col
        return None
    
    def _find_volume_column(self, df: pd.DataFrame, suffix: str) -> Optional[str]:
        """Find volume column with given suffix."""
        volume_candidates = ['Volume', 'volume', 'Vol', 'vol']
        for col in df.columns:
            if suffix in col:
                for candidate in volume_candidates:
                    if candidate in col:
                        return col
        return None
    
    def get_performance_summary(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance summary from merged data."""
        if merged_data.empty:
            return {}
        
        try:
            summary = {
                'total_stocks': len(merged_data),
                'gainers': len(merged_data[merged_data['Price_Change_Pct'] > 0]),
                'losers': len(merged_data[merged_data['Price_Change_Pct'] < 0]),
                'neutral': len(merged_data[merged_data['Price_Change_Pct'] == 0]),
                'avg_change': merged_data['Price_Change_Pct'].mean(),
                'max_gain': merged_data['Price_Change_Pct'].max(),
                'max_loss': merged_data['Price_Change_Pct'].min(),
                'median_change': merged_data['Price_Change_Pct'].median(),
                'std_deviation': merged_data['Price_Change_Pct'].std()
            }
            return summary
        except Exception as e:
            st.warning(f"Error generating performance summary: {str(e)}")
            return {}
    
    def get_top_performers(self, merged_data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top performing stocks."""
        if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
            return pd.DataFrame()
        
        try:
            return merged_data.nlargest(n, 'Price_Change_Pct')
        except Exception as e:
            st.warning(f"Error getting top performers: {str(e)}")
            return pd.DataFrame()
    
    def get_bottom_performers(self, merged_data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get bottom performing stocks."""
        if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
            return pd.DataFrame()
        
        try:
            return merged_data.nsmallest(n, 'Price_Change_Pct')
        except Exception as e:
            st.warning(f"Error getting bottom performers: {str(e)}")
            return pd.DataFrame()