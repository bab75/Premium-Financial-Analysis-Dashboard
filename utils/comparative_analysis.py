"""
Comparative Analysis Module for Phase 1 Analysis
Performs simplified comparison between current and previous stock data, focusing on price calculations
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional

class ComparativeAnalysis:
    """Perform simplified comparative analysis between current and previous stock data."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._prepare_comparative_data()
    
    def _prepare_comparative_data(self):
        """Prepare merged dataset for comparative analysis with flexible column matching."""
        try:
            # Find symbol column with flexible naming
            symbol_col_current = self._find_symbol_column(self.current_data)
            symbol_col_previous = self._find_symbol_column(self.previous_data)
            
            if not symbol_col_current or not symbol_col_previous:
                st.error("Symbol column not found. Please ensure your data has a column named Symbol, Ticker, or similar")
                st.write(f"Current columns: {list(self.current_data.columns)}")
                st.write(f"Previous columns: {list(self.previous_data.columns)}")
                return
            
            # Standardize symbol column names
            current_data_clean = self.current_data.copy()
            previous_data_clean = self.previous_data.copy()
            
            if symbol_col_current != 'Symbol':
                current_data_clean = current_data_clean.rename(columns={symbol_col_current: 'Symbol'})
            if symbol_col_previous != 'Symbol':
                previous_data_clean = previous_data_clean.rename(columns={symbol_col_previous: 'Symbol'})
            
            # Clean and standardize symbols
            current_data_clean['Symbol'] = current_data_clean['Symbol'].astype(str).str.strip().str.upper()
            previous_data_clean['Symbol'] = previous_data_clean['Symbol'].astype(str).str.strip().str.upper()
            
            # Remove empty symbols
            current_data_clean = current_data_clean[current_data_clean['Symbol'].notna() & (current_data_clean['Symbol'] != '')]
            previous_data_clean = previous_data_clean[previous_data_clean['Symbol'].notna() & (previous_data_clean['Symbol'] != '')]
            
            # Debug: Show symbol counts and samples
            st.write(f"Current data symbols: {len(current_data_clean['Symbol'].unique())}")
            st.write(f"Previous data symbols: {len(previous_data_clean['Symbol'].unique())}")
            st.write(f"Sample current symbols: {current_data_clean['Symbol'].head().tolist()}")
            st.write(f"Sample previous symbols: {previous_data_clean['Symbol'].head().tolist()}")
            
            # Merge datasets on Symbol
            merged = pd.merge(
                current_data_clean, 
                previous_data_clean, 
                on='Symbol', 
                how='inner',
                suffixes=('_current', '_previous')
            )
            
            if merged.empty:
                st.warning(f"No matching symbols found. Current: {len(current_data_clean)}, Previous: {len(previous_data_clean)}")
                return
            
            # Calculate price changes
            price_col_current = self._find_price_column(merged, '_current')
            price_col_previous = self._find_price_column(merged, '_previous')
            
            if price_col_current and price_col_previous:
                # Debug: Show raw price column data before cleaning
                st.write(f"Selected price columns: {price_col_current}, {price_col_previous}")
                st.write(f"Sample raw {price_col_current} values: {merged[price_col_current].head().tolist()}")
                st.write(f"Sample raw {price_col_previous} values: {merged[price_col_previous].head().tolist()}")
                
                # Convert price columns to numeric
                merged[price_col_current] = self._clean_numeric_column(merged[price_col_current])
                merged[price_col_previous] = self._clean_numeric_column(merged[price_col_previous])
                
                # Debug: Show cleaned price values
                st.write(f"Sample cleaned {price_col_current} values: {merged[price_col_current].head().tolist()}")
                st.write(f"Sample cleaned {price_col_previous} values: {merged[price_col_previous].head().tolist()}")
                
                # Check for valid numeric data
                valid_prices = merged[merged[price_col_current].notna() & merged[price_col_previous].notna()]
                st.write(f"Number of stocks with valid prices: {len(valid_prices)}")
                
                if len(valid_prices) == 0:
                    st.warning("No valid numeric price data found after cleaning.")
                    return
                
                # Calculate profit/loss and percentage change
                merged['Profit_Loss'] = merged[price_col_current] - merged[price_col_previous]
                merged['Price_Change_Pct'] = ((merged['Profit_Loss'] / 
                                             merged[price_col_previous].replace(0, np.nan)) * 100).round(2)
                
                # Add profit/loss classification
                merged['Profit_Loss_Status'] = merged['Price_Change_Pct'].apply(
                    lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                )
            else:
                st.warning(f"Price columns not found. Available columns: {list(merged.columns)}")
                return
            
            self.merged_data = merged
            st.success(f"Successfully merged {len(merged)} stocks for comparative analysis")
            
        except Exception as e:
            st.error(f"Error preparing comparative data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    def _find_symbol_column(self, df: pd.DataFrame) -> str:
        """Find symbol column with flexible naming."""
        possible_names = [
            'Symbol', 'symbol', 'SYMBOL', 
            'Ticker', 'ticker', 'TICKER', 
            'Stock', 'stock', 'STOCK',
            'Code', 'code', 'CODE'
        ]
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in ['symbol', 'ticker', 'stock', 'code']):
                return col
        
        if len(df.columns) > 0:
            st.warning(f"No symbol column found, using first column: {df.columns[0]}")
            return df.columns[0]
        
        return None
    
    def _find_price_column(self, df: pd.DataFrame, suffix: str) -> str:
        """Find price column with flexible naming."""
        possible_names = [
            f'Last Sale{suffix}', f'Price{suffix}', f'Close{suffix}', 
            f'Last Price{suffix}', f'Current Price{suffix}', f'Market Price{suffix}',
            f'last sale{suffix}', f'price{suffix}', f'close{suffix}',
            f'LAST SALE{suffix}', f'PRICE{suffix}', f'CLOSE{suffix}'
        ]
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['price', 'sale', 'close']) and suffix.lower() in col_lower:
                return col
        
        return None
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric values."""
        try:
            if series.dtype in ['object', 'string']:
                cleaned = series.astype(str).str.strip()
                # Handle common non-numeric cases
                cleaned = cleaned.replace(['', 'N/A', 'NA', 'n/a', '-', 'None'], np.nan)
                cleaned = cleaned.str.replace(r'[$,€£¥₹%]', '', regex=True)  # Remove currency and % symbols
                cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)    # Keep only digits, decimal, and negative
                cleaned = pd.to_numeric(cleaned, errors='coerce')
            else:
                cleaned = pd.to_numeric(series, errors='coerce')
            # Debug: Report non-numeric values
            if cleaned.isna().sum() > 0:
                st.warning(f"Found {cleaned.isna().sum()} non-numeric or missing values in column")
            return cleaned
        except Exception as e:
            st.error(f"Error cleaning numeric column: {str(e)}")
            return series
    
    def get_performance_summary(self) -> Dict:
        """Generate simplified performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            if 'Price_Change_Pct' in self.merged_data.columns:
                price_changes = self.merged_data['Price_Change_Pct'].dropna()
                if not price_changes.empty:
                    gainers = int((price_changes > 0).sum())
                    losers = int((price_changes < 0).sum())
                    unchanged = int((price_changes == 0).sum())
                    avg_change = float(price_changes.mean()) if not price_changes.empty else 0.0
                    max_gain = float(price_changes.max()) if not price_changes.empty else 0.0
                    min_loss = float(price_changes.min()) if not price_changes.empty else 0.0
                    
                    summary = {
                        'total_stocks': len(self.merged_data),
                        'gainers': gainers,
                        'losers': losers,
                        'unchanged': unchanged,
                        'avg_change': avg_change,
                        'max_gain': max_gain,
                        'min_loss': min_loss
                    }
                    return summary
            
            st.warning("No valid Price_Change_Pct data available for summary")
            return {
                'total_stocks': len(self.merged_data),
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change': 0.0,
                'max_gain': 0.0,
                'min_loss': 0.0
            }
            
        except Exception as e:
            st.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def get_calculated_data(self) -> pd.DataFrame:
        """Return the merged DataFrame with calculated metrics."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        return self.merged_data.copy()
