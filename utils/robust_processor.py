"""
Robust Data Processor - Final Solution for All Data Processing Issues
Fixes column matching, data type conversion, and comparison logic
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import re

class RobustProcessor:
    """Comprehensive data processor with bulletproof error handling."""
    
    def __init__(self):
        self.column_mappings = {
            'symbol': ['symbol', 'ticker', 'stock', 'code', 'stock_symbol', 'company_symbol'],
            'name': ['name', 'company_name', 'company', 'stock_name', 'security_name', 'description'],
            'price': ['last_sale', 'price', 'last_price', 'close_price', 'current_price', 'last', 'close', 'closing_price'],
            'change_pct': ['%_change', 'percent_change', 'pct_change', 'change_percent', 'percentage_change', 'pct_chg'],
            'change_amt': ['net_change', 'change', 'price_change', 'net_chg', 'chg', 'change_amount'],
            'volume': ['volume', 'trade_volume', 'trading_volume', 'vol', 'shares_traded'],
            'market_cap': ['market_cap', 'marketcap', 'market_capitalization', 'cap', 'mkt_cap'],
            'sector': ['sector', 'industry_sector', 'business_sector', 'gics_sector'],
            'industry': ['industry', 'sub_industry', 'business_industry', 'gics_industry'],
            'country': ['country', 'nation', 'location', 'domicile', 'headquarters']
        }
    
    def process_uploaded_data(self, file) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Process uploaded file with comprehensive error handling."""
        try:
            # Read file based on extension
            if file.name.endswith('.csv'):
                df = self._read_csv_robust(file)
            else:
                df = pd.read_excel(file)
            
            if df is None or df.empty:
                return None, {'error': 'Could not read file or file is empty'}
            
            # Initial file info
            st.info(f"Loaded file: {len(df)} rows, {len(df.columns)} columns")
            
            # Show raw column names for debugging
            st.info(f"Detected columns: {list(df.columns)}")
            
            # Map columns to standard names
            mapped_df, mapping_report = self._map_columns_intelligent(df)
            
            if mapped_df.empty:
                return None, {'error': 'No recognizable data columns found', 'mapping_report': mapping_report}
            
            # Clean and validate data
            cleaned_df, quality_report = self._clean_and_validate(mapped_df)
            
            # Combine reports
            final_report = {
                'total_rows': len(df),
                'processed_rows': len(cleaned_df),
                'column_mapping': mapping_report,
                'quality_metrics': quality_report
            }
            
            return cleaned_df, final_report
            
        except Exception as e:
            st.error(f"File processing error: {str(e)}")
            return None, {'error': str(e)}
    
    def _read_csv_robust(self, file) -> Optional[pd.DataFrame]:
        """Read CSV with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=encoding, sep=separator)
                    if len(df.columns) > 1 and len(df) > 0:
                        return df
                except:
                    continue
        
        return None
    
    def _map_columns_intelligent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Intelligently map columns to standard names."""
        mapping_results = {}
        mapped_df = df.copy()
        
        # Clean column names first
        cleaned_columns = {}
        for col in df.columns:
            clean_col = str(col).lower().strip().replace(' ', '_').replace('%', 'pct')
            cleaned_columns[col] = clean_col
        
        # Find mappings
        for standard_name, variations in self.column_mappings.items():
            found_column = None
            
            # Try exact matches first
            for orig_col, clean_col in cleaned_columns.items():
                if clean_col in variations or any(var in clean_col for var in variations):
                    found_column = orig_col
                    break
            
            # Try partial matches
            if not found_column:
                for orig_col, clean_col in cleaned_columns.items():
                    for variation in variations:
                        if variation in clean_col or clean_col in variation:
                            found_column = orig_col
                            break
                    if found_column:
                        break
            
            if found_column:
                mapping_results[standard_name] = found_column
                # Rename column in dataframe
                standard_col_name = standard_name.replace('_', ' ').title()
                if standard_name == 'change_pct':
                    standard_col_name = '% Change'
                elif standard_name == 'change_amt':
                    standard_col_name = 'Net Change'
                elif standard_name == 'price':
                    standard_col_name = 'Last Sale'
                elif standard_name == 'market_cap':
                    standard_col_name = 'Market Cap'
                
                mapped_df = mapped_df.rename(columns={found_column: standard_col_name})
        
        return mapped_df, mapping_results
    
    def _clean_and_validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and validate the mapped dataframe."""
        quality_metrics = {}
        cleaned_df = df.copy()
        
        # Clean Symbol column
        if 'Symbol' in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df['Symbol'] = cleaned_df['Symbol'].astype(str).str.strip().str.upper()
            cleaned_df = cleaned_df[cleaned_df['Symbol'].notna()]
            cleaned_df = cleaned_df[cleaned_df['Symbol'] != '']
            cleaned_df = cleaned_df[cleaned_df['Symbol'] != 'NAN']
            quality_metrics['valid_symbols'] = len(cleaned_df)
            quality_metrics['symbol_cleanup_removed'] = original_count - len(cleaned_df)
        
        # Clean price columns
        price_columns = ['Last Sale', 'Net Change']
        for col in price_columns:
            if col in cleaned_df.columns:
                original_valid = cleaned_df[col].notna().sum()
                cleaned_df[col] = self._clean_price_column(cleaned_df[col])
                final_valid = cleaned_df[col].notna().sum()
                quality_metrics[f'{col}_conversion'] = {
                    'original_valid': int(original_valid),
                    'final_valid': int(final_valid)
                }
        
        # Clean percentage column
        if '% Change' in cleaned_df.columns:
            original_valid = cleaned_df['% Change'].notna().sum()
            cleaned_df['% Change'] = self._clean_percentage_column(cleaned_df['% Change'])
            final_valid = cleaned_df['% Change'].notna().sum()
            quality_metrics['percent_change_conversion'] = {
                'original_valid': int(original_valid),
                'final_valid': int(final_valid)
            }
        
        # Clean volume column
        if 'Volume' in cleaned_df.columns:
            cleaned_df['Volume'] = self._clean_volume_column(cleaned_df['Volume'])
        
        # Clean market cap column
        if 'Market Cap' in cleaned_df.columns:
            cleaned_df['Market Cap'] = self._clean_market_cap_column(cleaned_df['Market Cap'])
        
        # Fill missing categorical data
        categorical_cols = ['Name', 'Sector', 'Industry', 'Country']
        for col in categorical_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
        
        # Remove rows with no price data
        if 'Last Sale' in cleaned_df.columns:
            before_price_filter = len(cleaned_df)
            cleaned_df = cleaned_df[cleaned_df['Last Sale'].notna()]
            after_price_filter = len(cleaned_df)
            quality_metrics['price_filter_removed'] = before_price_filter - after_price_filter
        
        return cleaned_df, quality_metrics
    
    def _clean_price_column(self, series: pd.Series) -> pd.Series:
        """Clean price column removing $ signs and converting to float."""
        def clean_price(val):
            if pd.isna(val):
                return np.nan
            try:
                # Convert to string and clean
                val_str = str(val).strip()
                # Remove currency symbols and commas
                val_str = re.sub(r'[$,€£¥₹]', '', val_str)
                # Handle parentheses (negative values)
                if '(' in val_str and ')' in val_str:
                    val_str = '-' + val_str.replace('(', '').replace(')', '')
                return float(val_str)
            except:
                return np.nan
        
        return series.apply(clean_price)
    
    def _clean_percentage_column(self, series: pd.Series) -> pd.Series:
        """Clean percentage column."""
        def clean_percentage(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                # Remove % sign
                val_str = val_str.replace('%', '')
                # Handle parentheses (negative values)
                if '(' in val_str and ')' in val_str:
                    val_str = '-' + val_str.replace('(', '').replace(')', '')
                return float(val_str)
            except:
                return np.nan
        
        return series.apply(clean_percentage)
    
    def _clean_volume_column(self, series: pd.Series) -> pd.Series:
        """Clean volume column."""
        def clean_volume(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip().replace(',', '')
                return float(val_str)
            except:
                return np.nan
        
        return series.apply(clean_volume)
    
    def _clean_market_cap_column(self, series: pd.Series) -> pd.Series:
        """Clean market cap column handling M, B, K suffixes."""
        def clean_market_cap(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip().upper()
                # Remove $ and commas
                val_str = re.sub(r'[$,]', '', val_str)
                
                # Handle suffixes
                multiplier = 1
                if val_str.endswith('K'):
                    multiplier = 1000
                    val_str = val_str[:-1]
                elif val_str.endswith('M'):
                    multiplier = 1000000
                    val_str = val_str[:-1]
                elif val_str.endswith('B'):
                    multiplier = 1000000000
                    val_str = val_str[:-1]
                
                return float(val_str) * multiplier
            except:
                return np.nan
        
        return series.apply(clean_market_cap)
    
    def compare_datasets(self, current_data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """Compare two datasets and calculate changes."""
        try:
            # Ensure both datasets have Symbol column
            if 'Symbol' not in current_data.columns or 'Symbol' not in previous_data.columns:
                st.error("Both datasets must have a Symbol column for comparison")
                return pd.DataFrame()
            
            # Clean symbols for matching
            current_clean = current_data.copy()
            previous_clean = previous_data.copy()
            
            current_clean['Symbol'] = current_clean['Symbol'].astype(str).str.strip().str.upper()
            previous_clean['Symbol'] = previous_clean['Symbol'].astype(str).str.strip().str.upper()
            
            # Remove invalid symbols
            current_clean = current_clean[current_clean['Symbol'].notna() & (current_clean['Symbol'] != '') & (current_clean['Symbol'] != 'NAN')]
            previous_clean = previous_clean[previous_clean['Symbol'].notna() & (previous_clean['Symbol'] != '') & (previous_clean['Symbol'] != 'NAN')]
            
            if current_clean.empty or previous_clean.empty:
                st.error("No valid symbols found in one or both datasets")
                return pd.DataFrame()
            
            # Display matching info
            current_symbols = set(current_clean['Symbol'].unique())
            previous_symbols = set(previous_clean['Symbol'].unique())
            common_symbols = current_symbols.intersection(previous_symbols)
            
            st.info(f"Current dataset: {len(current_symbols)} unique symbols")
            st.info(f"Previous dataset: {len(previous_symbols)} unique symbols")
            st.info(f"Common symbols: {len(common_symbols)}")
            
            if len(common_symbols) == 0:
                st.error("No matching symbols found between datasets")
                st.info(f"Sample current symbols: {list(current_symbols)[:10]}")
                st.info(f"Sample previous symbols: {list(previous_symbols)[:10]}")
                return pd.DataFrame()
            
            # Merge datasets
            merged = pd.merge(
                current_clean, 
                previous_clean, 
                on='Symbol', 
                how='inner',
                suffixes=('_current', '_previous')
            )
            
            # Calculate price changes if both datasets have price data
            if 'Last Sale_current' in merged.columns and 'Last Sale_previous' in merged.columns:
                merged['Price_Change'] = merged['Last Sale_current'] - merged['Last Sale_previous']
                # Fix percentage calculation formula
                merged['Price_Change_Pct'] = (
                    merged['Price_Change'] / merged['Last Sale_previous'].replace(0, np.nan) * 100
                ).round(2)
                
                # Validate calculations
                valid_changes = merged['Price_Change_Pct'].notna().sum()
                st.success(f"Successfully calculated price changes for {valid_changes} stocks")
            else:
                st.warning("Price data not found in both datasets for comparison")
            
            return merged
            
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_summary(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance summary from merged data."""
        if merged_data.empty or 'Price_Change_Pct' not in merged_data.columns:
            return {}
        
        valid_data = merged_data['Price_Change_Pct'].notna()
        changes = merged_data.loc[valid_data, 'Price_Change_Pct']
        
        if len(changes) == 0:
            return {}
        
        return {
            'total_stocks': len(changes),
            'avg_change': float(changes.mean()),
            'gainers': int((changes > 0).sum()),
            'losers': int((changes < 0).sum()),
            'unchanged': int((changes == 0).sum()),
            'max_gain': float(changes.max()),
            'max_loss': float(changes.min()),
            'std_dev': float(changes.std())
        }