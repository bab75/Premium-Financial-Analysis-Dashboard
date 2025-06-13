"""
Comparative Analysis Module for Phase 1 Analysis
Performs comprehensive comparison between current and previous stock data
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta

class ComparativeAnalysis:
    """Perform comprehensive comparative analysis between current and previous stock data."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._prepare_comparative_data()
    
    def _prepare_comparative_data(self):
        """Prepare merged dataset for comparative analysis with flexible column matching."""
        try:
            symbol_col_current = self._find_symbol_column(self.current_data)
            symbol_col_previous = self._find_symbol_column(self.previous_data)
            
            if not symbol_col_current or not symbol_col_previous:
                st.error(f"Symbol column not found. Please ensure your data has a column named Symbol, Ticker, or similar")
                return
            
            current_data_clean = self.current_data.copy()
            previous_data_clean = self.previous_data.copy()
            
            if symbol_col_current != 'Symbol':
                current_data_clean = current_data_clean.rename(columns={symbol_col_current: 'Symbol'})
            if symbol_col_previous != 'Symbol':
                previous_data_clean = previous_data_clean.rename(columns={symbol_col_previous: 'Symbol'})
            
            current_data_clean['Symbol'] = current_data_clean['Symbol'].astype(str).str.strip().str.upper()
            previous_data_clean['Symbol'] = previous_data_clean['Symbol'].astype(str).str.strip().str.upper()
            
            current_data_clean = current_data_clean[current_data_clean['Symbol'].notna() & (current_data_clean['Symbol'] != '')]
            previous_data_clean = previous_data_clean[previous_data_clean['Symbol'].notna() & (previous_data_clean['Symbol'] != '')]
            
            # Remove duplicates
            current_data_clean = current_data_clean.drop_duplicates(subset=['Symbol'], keep='first')
            previous_data_clean = previous_data_clean.drop_duplicates(subset=['Symbol'], keep='first')
            
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
            
            change_pct_col = None
            change_col = None
            
            for col in merged.columns:
                if 'Change_current' in col and '%' in col:
                    change_pct_col = col
                elif 'Net Change_current' in col:
                    change_col = col
            
            if change_pct_col:
                st.info(f"Using existing change data from column: {change_pct_col}")
                merged['Price_Change_Pct'] = self._clean_numeric_column(merged[change_pct_col])
                
                merged['Profit_Loss'] = merged['Price_Change_Pct'].apply(
                    lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                )
                
                if change_col:
                    merged['Price_Change'] = self._clean_numeric_column(merged[change_col])
            else:
                price_col_current = self._find_price_column(merged, '_current')
                price_col_previous = self._find_price_column(merged, '_previous')
                
                if price_col_current and price_col_previous:
                    merged[price_col_current] = self._clean_numeric_column(merged[price_col_current])
                    merged[price_col_previous] = self._clean_numeric_column(merged[price_col_previous])
                    
                    merged['Price_Change'] = merged[price_col_current] - merged[price_col_previous]
                    merged['Price_Change_Pct'] = ((merged[price_col_current] - merged[price_col_previous]) / 
                                                merged[price_col_previous].replace(0, np.nan)) * 100
                    
                    merged['Profit_Loss'] = merged['Price_Change_Pct'].apply(
                        lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                    )
                else:
                    st.warning(f"Neither change columns nor price columns found. Available columns: {list(merged.columns)}")
                    return
            
            volume_col_current = self._find_volume_column(merged, '_current')
            volume_col_previous = self._find_volume_column(merged, '_previous')
            
            if volume_col_current and volume_col_previous:
                merged[volume_col_current] = self._clean_numeric_column(merged[volume_col_current])
                merged[volume_col_previous] = self._clean_numeric_column(merged[volume_col_previous])
                
                merged['Volume_Change'] = merged[volume_col_current] - merged[volume_col_previous]
                merged['Volume_Change_Pct'] = ((merged[volume_col_current] - merged[volume_col_previous]) / 
                                             merged[volume_col_previous].replace(0, np.nan)) * 100
            
            mcap_col_current = self._find_market_cap_column(merged, '_current')
            mcap_col_previous = self._find_market_cap_column(merged, '_previous')
            
            if mcap_col_current and mcap_col_previous:
                merged[mcap_col_current] = self._clean_numeric_column(merged[mcap_col_current])
                merged[mcap_col_previous] = self._clean_numeric_column(merged[mcap_col_previous])
                
                merged['MarketCap_Change'] = merged[mcap_col_current] - merged[mcap_col_previous]
                merged['MarketCap_Change_Pct'] = ((merged[mcap_col_current] - merged[mcap_col_previous]) / 
                                                merged[mcap_col_previous].replace(0, np.nan)) * 100
            
            # Remove rows with NaN Price_Change_Pct
            initial_count = len(merged)
            merged = merged.dropna(subset=['Price_Change_Pct'])
            if len(merged) < initial_count:
                st.warning(f"Removed {initial_count - len(merged)} rows with invalid Price_Change_Pct values")
            
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
            'Code', 'code', 'CODE',
            'Name', 'name', 'NAME',
            'Company', 'company', 'COMPANY',
            'Security', 'security', 'SECURITY'
        ]
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in ['symbol', 'ticker', 'stock', 'code', 'name', 'company']):
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
            f'LAST SALE{suffix}', f'PRICE{suffix}', f'CLOSE{suffix}',
            f'Net Change{suffix}', f'% Change{suffix}', f'Change{suffix}'
        ]
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['price', 'sale', 'close', 'value']) and suffix.lower() in col_lower:
                return col
        
        return None
    
    def _find_volume_column(self, df: pd.DataFrame, suffix: str) -> str:
        """Find volume column with flexible naming."""
        possible_names = [f'Volume{suffix}', f'Vol{suffix}', f'Trading Volume{suffix}']
        for col in possible_names:
            if col in df.columns:
                return col
        return None
    
    def _find_market_cap_column(self, df: pd.DataFrame, suffix: str) -> str:
        """Find market cap column with flexible naming."""
        possible_names = [
            f'Market Cap{suffix}', f'MarketCap{suffix}', f'Market Capitalization{suffix}',
            f'Mkt Cap{suffix}', f'Market Value{suffix}'
        ]
        for col in possible_names:
            if col in df.columns:
                return col
        return None
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric values."""
        try:
            if series.dtype in ['object', 'string']:
                cleaned = series.astype(str).str.strip()
                
                is_percentage = cleaned.str.contains('%', na=False).any()
                
                if is_percentage:
                    cleaned = cleaned.str.replace('%', '', regex=False)
                    cleaned = cleaned.str.replace(',', '', regex=False)
                    cleaned = pd.to_numeric(cleaned, errors='coerce')
                else:
                    cleaned = cleaned.str.replace(r'[$,€£¥₹]', '', regex=True)
                    cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
                    cleaned = pd.to_numeric(cleaned, errors='coerce')
            else:
                cleaned = pd.to_numeric(series, errors='coerce')
            
            if cleaned.isna().all():
                st.warning(f"All values in column {series.name} could not be converted to numeric")
            
            return cleaned
        except Exception as e:
            st.error(f"Error cleaning numeric column {series.name}: {str(e)}")
            return pd.Series(np.nan, index=series.index)
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            summary = {
                'total_stocks': len(self.merged_data),
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change': 0.0,
                'max_gain': 0.0,
                'max_loss': 0.0
            }
            
            if 'Price_Change_Pct' in self.merged_data.columns:
                price_changes = self.merged_data['Price_Change_Pct'].dropna()
                if not price_changes.empty:
                    summary['gainers'] = int((price_changes > 0).sum())
                    summary['losers'] = int((price_changes < 0).sum())
                    summary['unchanged'] = int((price_changes == 0).sum())
                    summary['avg_change'] = float(price_changes.mean())
                    summary['max_gain'] = float(price_changes.max())
                    summary['max_loss'] = float(price_changes.min())
            
            return summary
        except Exception as e:
            st.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Analyze performance by sector."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            sector_col = 'Sector_current' if 'Sector_current' in self.merged_data.columns else None
            if not sector_col:
                return pd.DataFrame()
            
            sector_analysis = self.merged_data.groupby(sector_col).agg({
                'Price_Change_Pct': ['mean', 'median', 'std', 'count'],
                'MarketCap_Change_Pct': ['mean', 'median'] if 'MarketCap_Change_Pct' in self.merged_data.columns else ['count'],
                'Volume_Change_Pct': ['mean', 'median'] if 'Volume_Change_Pct' in self.merged_data.columns else ['count']
            }).round(2)
            
            sector_analysis.columns = ['_'.join(col).strip() for col in sector_analysis.columns]
            sector_analysis = sector_analysis.reset_index()
            
            return sector_analysis
        except Exception as e:
            st.error(f"Error in sector analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_industry_analysis(self) -> pd.DataFrame:
        """Analyze performance by industry (top 20)."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            industry_col = 'Industry_current' if 'Industry_current' in self.merged_data.columns else None
            if not industry_col:
                return pd.DataFrame()
            
            industry_analysis = self.merged_data.groupby(industry_col).agg({
                'Price_Change_Pct': ['mean', 'median', 'std', 'count'],
                'MarketCap_Change_Pct': ['mean'] if 'MarketCap_Change_Pct' in self.merged_data.columns else ['count'],
                'Volume_Change_Pct': ['mean'] if 'Volume_Change_Pct' in self.merged_data.columns else ['count']
            }).round(2)
            
            industry_analysis.columns = ['_'.join(col).strip() for col in industry_analysis.columns]
            industry_analysis = industry_analysis.reset_index()
            
            if 'Price_Change_Pct_mean' in industry_analysis.columns:
                industry_analysis = industry_analysis.sort_values('Price_Change_Pct_mean', ascending=False).head(20)
            
            return industry_analysis
        except Exception as e:
            st.error(f"Error in industry analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_country_analysis(self) -> pd.DataFrame:
        """Analyze performance by country."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            country_col = 'Country_current' if 'Country_current' in self.merged_data.columns else None
            if not country_col:
