"""
Fixed Data Processor with Comprehensive Error Handling
Addresses all data processing issues and ensures proper functionality
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Tuple, Any

class FixedDataProcessor:
    """Fixed data processor with proper error handling and data validation."""
    
    def __init__(self):
        self.required_daily_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change', 'Market Cap', 'Volume', 'Sector', 'Industry']
    
    def process_comparative_data(self, current_data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """Process and merge datasets for comparative analysis with proper error handling."""
        try:
            # Find symbol columns
            current_symbol_col = self._find_symbol_column(current_data)
            previous_symbol_col = self._find_symbol_column(previous_data)
            
            if not current_symbol_col or not previous_symbol_col:
                st.error("Symbol column not found in one or both datasets")
                return pd.DataFrame()
            
            # Prepare datasets
            current_clean = self._prepare_dataset(current_data, current_symbol_col, "_current")
            previous_clean = self._prepare_dataset(previous_data, previous_symbol_col, "_previous")
            
            if current_clean.empty or previous_clean.empty:
                st.error("One or both datasets are empty after cleaning")
                return pd.DataFrame()
            
            # Merge datasets
            merged = pd.merge(current_clean, previous_clean, on='Symbol', how='inner')
            
            if merged.empty:
                st.error("No matching symbols found between datasets")
                return pd.DataFrame()
            
            # Calculate changes
            merged = self._calculate_changes(merged)
            
            return merged
            
        except Exception as e:
            st.error(f"Error processing comparative data: {str(e)}")
            return pd.DataFrame()
    
    def _find_symbol_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find symbol column in dataset."""
        symbol_candidates = ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'Stock', 'stock']
        for col in symbol_candidates:
            if col in df.columns:
                return col
        return None
    
    def _prepare_dataset(self, df: pd.DataFrame, symbol_col: str, suffix: str) -> pd.DataFrame:
        """Prepare dataset for merging."""
        try:
            # Create a copy
            df_clean = df.copy()
            
            # Rename symbol column to standard name
            if symbol_col != 'Symbol':
                df_clean = df_clean.rename(columns={symbol_col: 'Symbol'})
            
            # Clean symbol column
            df_clean['Symbol'] = df_clean['Symbol'].astype(str).str.strip()
            df_clean = df_clean[df_clean['Symbol'] != '']
            df_clean = df_clean[df_clean['Symbol'] != 'nan']
            df_clean = df_clean[df_clean['Symbol'].notna()]
            
            # Find and process price column
            price_col = self._find_price_column(df)
            if price_col:
                df_clean[f'Price{suffix}'] = pd.to_numeric(df_clean[price_col], errors='coerce')
            
            # Find and process volume column
            volume_col = self._find_volume_column(df)
            if volume_col:
                df_clean[f'Volume{suffix}'] = pd.to_numeric(df_clean[volume_col], errors='coerce')
            
            # Keep other relevant columns
            keep_columns = ['Symbol']
            if f'Price{suffix}' in df_clean.columns:
                keep_columns.append(f'Price{suffix}')
            if f'Volume{suffix}' in df_clean.columns:
                keep_columns.append(f'Volume{suffix}')
            
            # Add sector and industry if available
            if 'Sector' in df_clean.columns:
                keep_columns.append('Sector')
            if 'Industry' in df_clean.columns:
                keep_columns.append('Industry')
            
            result = df_clean[keep_columns]
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error preparing dataset: {str(e)}")
            return pd.DataFrame()
    
    def _find_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find price column in dataset."""
        price_candidates = ['Last Sale', 'Price', 'Close', 'Last Price', 'Current Price']
        for col in price_candidates:
            if col in df.columns:
                return col
        return None
    
    def _find_volume_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find volume column in dataset."""
        volume_candidates = ['Volume', 'volume', 'VOLUME', 'Trading Volume']
        for col in volume_candidates:
            if col in df.columns:
                return col
        return None
    
    def _calculate_changes(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Calculate price and volume changes."""
        try:
            # Calculate price changes
            if 'Price_current' in merged.columns and 'Price_previous' in merged.columns:
                merged['Price_Change_Amount'] = merged['Price_current'] - merged['Price_previous']
                merged['Price_Change_Pct'] = ((merged['Price_current'] - merged['Price_previous']) / merged['Price_previous'] * 100).round(2)
            
            # Calculate volume changes
            if 'Volume_current' in merged.columns and 'Volume_previous' in merged.columns:
                merged['Volume_Change_Pct'] = ((merged['Volume_current'] - merged['Volume_previous']) / merged['Volume_previous'] * 100).round(2)
            
            return merged
            
        except Exception as e:
            st.error(f"Error calculating changes: {str(e)}")
            return merged
    
    def get_performance_summary(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance summary from merged data."""
        try:
            if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
                return {}
            
            valid_changes = merged_data['Price_Change_Pct'].dropna()
            
            if len(valid_changes) == 0:
                return {}
            
            summary = {
                'total_stocks': len(merged_data),
                'avg_change': float(valid_changes.mean()),
                'gainers': int((valid_changes > 0).sum()),
                'losers': int((valid_changes < 0).sum()),
                'max_gain': float(valid_changes.max()),
                'max_loss': float(valid_changes.min())
            }
            
            return summary
            
        except Exception as e:
            st.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def get_top_performers(self, merged_data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top performing stocks."""
        try:
            if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
                return pd.DataFrame()
            
            valid_data = merged_data.dropna(subset=['Price_Change_Pct'])
            if valid_data.empty:
                return pd.DataFrame()
            
            return valid_data.nlargest(n, 'Price_Change_Pct')
            
        except Exception as e:
            st.error(f"Error getting top performers: {str(e)}")
            return pd.DataFrame()
    
    def get_bottom_performers(self, merged_data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get bottom performing stocks."""
        try:
            if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
                return pd.DataFrame()
            
            valid_data = merged_data.dropna(subset=['Price_Change_Pct'])
            if valid_data.empty:
                return pd.DataFrame()
            
            return valid_data.nsmallest(n, 'Price_Change_Pct')
            
        except Exception as e:
            st.error(f"Error getting bottom performers: {str(e)}")
            return pd.DataFrame()