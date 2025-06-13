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
import re

class ComparativeAnalysis:
    """Perform comprehensive comparative analysis between current and previous stock data."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._prepare_comparative_data()
    
    def _prepare_comparative_data(self):
        """Prepare simplified dataset for comparative analysis with specific output columns."""
        try:
            # Find symbol column with flexible naming
            symbol_col_current = self._find_symbol_column(self.current_data)
            symbol_col_previous = self._find_symbol_column(self.previous_data)
            
            if not symbol_col_current or not symbol_col_previous:
                st.error("Symbol column not found. Please ensure your data has a column named Symbol, Ticker, or similar.")
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
            
            # Find common symbols using set intersection
            common_symbols = set(current_data_clean['Symbol']).intersection(set(previous_data_clean['Symbol']))
            output_data = pd.DataFrame({'Symbol': list(common_symbols)})
            
            if output_data.empty:
                st.warning(f"No matching symbols found. Current: {len(current_data_clean)}, Previous: {len(previous_data_clean)}")
                return
            
            # Merge with current and previous data to get required columns
            output_data = output_data.merge(current_data_clean, on='Symbol', how='inner')
            output_data = output_data.merge(previous_data_clean, on='Symbol', how='inner', suffixes=('_curr', '_prev'))
            
            # Find price columns
            price_col_curr = self._find_price_column(current_data_clean)
            price_col_prev = self._find_price_column(previous_data_clean)
            
            if not price_col_curr or not price_col_prev:
                st.error(f"Price columns not found. Current: {price_col_curr}, Previous: {price_col_prev}")
                return
            
            st.info(f"Price columns identified - Current: {price_col_curr}, Previous: {price_col_prev}")
            
            # Clean and assign price columns
            output_data['Last Sale_curr'] = self._clean_numeric_column(output_data[f'{price_col_curr}_curr'])
            output_data['Last Sale_prev'] = self._clean_numeric_column(output_data[f'{price_col_prev}_prev'])
            
            # Log invalid Last Sale values
            invalid_curr = output_data['Last Sale_curr'].isna().sum()
            invalid_prev = output_data['Last Sale_prev'].isna().sum()
            if invalid_curr > 0 or invalid_prev > 0:
                st.warning(f"Invalid price values detected: Current ({invalid_curr}), Previous ({invalid_prev})")
            
            # Find net change columns
            net_change_col_curr = self._find_net_change_column(current_data_clean)
            net_change_col_prev = self._find_net_change_column(previous_data_clean)
            
            if net_change_col_curr:
                output_data['Net Change_curr'] = self._clean_numeric_column(output_data[f'{net_change_col_curr}_curr'])
            else:
                output_data['Net Change_curr'] = output_data['Last Sale_curr'] - output_data['Last Sale_prev']
            
            if net_change_col_prev:
                output_data['Net Change_prev'] = self._clean_numeric_column(output_data[f'{net_change_col_prev}_prev'])
            else:
                output_data['Net Change_prev'] = np.nan
            
            # Find % change columns
            pct_change_col_curr = self._find_pct_change_column(current_data_clean)
            pct_change_col_prev = self._find_pct_change_column(previous_data_clean)
            
            if pct_change_col_curr:
                output_data['% Change_curr'] = self._clean_numeric_column(output_data[f'{pct_change_col_curr}_curr'])
            else:
                output_data['% Change_curr'] = np.nan
            
            if pct_change_col_prev:
                output_data['% Change_prev'] = self._clean_numeric_column(output_data[f'{pct_change_col_prev}_prev'])
            else:
                output_data['% Change_prev'] = np.nan
            
            # Calculate % Change_calc
            valid_mask = (output_data['Last Sale_prev'].notna()) & (output_data['Last Sale_prev'] != 0)
            output_data['% Change_calc'] = np.nan
            output_data.loc[valid_mask, '% Change_calc'] = ((output_data.loc[valid_mask, 'Last Sale_curr'] - 
                                                            output_data.loc[valid_mask, 'Last Sale_prev']) / 
                                                           output_data.loc[valid_mask, 'Last Sale_prev']) * 100
            
            # Log invalid % Change_calc
            invalid_change = output_data['% Change_calc'].isna().sum()
            if invalid_change > 0:
                st.warning(f"{invalid_change} stocks have invalid % Change_calc due to missing or zero Last Sale_prev")
            
            # Add Profit/Loss classification
            output_data['Profit_Loss'] = output_data['% Change_calc'].apply(
                lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
            )
            
            # Calculate Profit/Loss Value
            quantity_col_curr = self._find_quantity_column(current_data_clean)
            if quantity_col_curr:
                output_data['Quantity'] = self._clean_numeric_column(output_data[f'{quantity_col_curr}_curr'])
                st.info(f"Using quantity data from column: {quantity_col_curr}")
            else:
                output_data['Quantity'] = 100
                st.info("No quantity column found, assuming 100 shares per stock")
            
            output_data['Profit_Loss_Value'] = np.nan
            output_data.loc[valid_mask, 'Profit_Loss_Value'] = (output_data.loc[valid_mask, 'Last Sale_curr'] - 
                                                                output_data.loc[valid_mask, 'Last Sale_prev']) * output_data.loc[valid_mask, 'Quantity']
            
            # Find volume columns
            volume_col_curr = self._find_volume_column(current_data_clean)
            volume_col_prev = self._find_volume_column(previous_data_clean)
            
            if volume_col_curr:
                output_data['Volume_curr'] = self._clean_numeric_column(output_data[f'{volume_col_curr}_curr'])
            else:
                output_data['Volume_curr'] = np.nan
            
            if volume_col_prev:
                output_data['Volume_prev'] = self._clean_numeric_column(output_data[f'{volume_col_prev}_prev'])
            else:
                output_data['Volume_prev'] = np.nan
            
            # Find sector, industry, country columns
            sector_col_curr = self._find_sector_column(current_data_clean)
            industry_col_curr = self._find_industry_column(current_data_clean)
            country_col_curr = self._find_country_column(current_data_clean)
            
            if sector_col_curr:
                output_data['Sector_curr'] = output_data[f'{sector_col_curr}_curr'].astype(str)
            else:
                output_data['Sector_curr'] = 'Unknown'
            
            if industry_col_curr:
                output_data['Industry_curr'] = output_data[f'{industry_col_curr}_curr'].astype(str)
            else:
                output_data['Industry_curr'] = 'Unknown'
            
            if country_col_curr:
                output_data['Country_curr'] = output_data[f'{country_col_curr}_curr'].astype(str)
            else:
                output_data['Country_curr'] = 'Unknown'
            
            # Select final columns
            final_columns = [
                'Symbol', 'Last Sale_prev', 'Net Change_prev', '% Change_prev',
                'Last Sale_curr', 'Net Change_curr', '% Change_curr',
                'Profit_Loss', '% Change_calc', 'Profit_Loss_Value',
                'Volume_prev', 'Volume_curr', 'Sector_curr', 'Industry_curr', 'Country_curr'
            ]
            self.merged_data = output_data[final_columns].round(2)
            
            st.success(f"Successfully prepared comparative analysis for {len(self.merged_data)} stocks")
            
        except Exception as e:
            st.error(f"Error preparing comparative data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    def _find_symbol_column(self, df: pd.DataFrame) -> str:
        """Find symbol column with flexible, case-insensitive naming."""
        possible_names = ['symbol', 'ticker', 'stock', 'code', 'name', 'company', 'security']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        if len(df.columns) > 0:
            st.warning(f"No symbol column found, using first column: {df.columns[0]}")
            return df.columns[0]
        
        return None
    
    def _find_price_column(self, df: pd.DataFrame) -> str:
        """Find price column with flexible, case-insensitive naming."""
        possible_names = ['last sale', 'price', 'close', 'last price', 'current price', 'market price', 'value']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_net_change_column(self, df: pd.DataFrame) -> str:
        """Find net change column with flexible naming."""
        possible_names = ['net change', 'change']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names) and '%' not in col_lower:
                return col
        
        return None
    
    def _find_pct_change_column(self, df: pd.DataFrame) -> str:
        """Find percentage change column with flexible naming."""
        possible_names = ['% change', 'change %', 'pct change', 'percent change']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_volume_column(self, df: pd.DataFrame) -> str:
        """Find volume column with flexible naming."""
        possible_names = ['volume', 'vol', 'trading volume']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_quantity_column(self, df: pd.DataFrame) -> str:
        """Find quantity column with flexible naming."""
        possible_names = ['quantity', 'shares', 'holdings']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_sector_column(self, df: pd.DataFrame) -> str:
        """Find sector column with flexible naming."""
        possible_names = ['sector']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_industry_column(self, df: pd.DataFrame) -> str:
        """Find industry column with flexible naming."""
        possible_names = ['industry']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _find_country_column(self, df: pd.DataFrame) -> str:
        """Find country column with flexible naming."""
        possible_names = ['country']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        return None
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric values, handling currency and percentages."""
        try:
            # Convert to string and clean
            cleaned = series.astype(str).str.strip()
            
            # Remove currency symbols, commas, and percentages
            cleaned = cleaned.replace(r'[\$,€£¥₹%]', '', regex=True).replace(',', '', regex=True)
            
            # Convert to numeric
            cleaned = pd.to_numeric(cleaned, errors='coerce')
            
            return cleaned
        except Exception as e:
            st.warning(f"Error cleaning numeric column: {str(e)}")
            return pd.Series(np.nan, index=series.index)
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            summary = {
                'total_stocks': len(self.merged_data),
                'gainers': int((self.merged_data['% Change_calc'] > 0).sum()),
                'losers': int((self.merged_data['% Change_calc'] < 0).sum()),
                'unchanged': int((self.merged_data['% Change_calc'] == 0).sum()),
                'avg_change': float(self.merged_data['% Change_calc'].mean()) if self.merged_data['% Change_calc'].notna().any() else np.nan,
                'max_gain': float(self.merged_data['% Change_calc'].max()) if self.merged_data['% Change_calc'].notna().any() else np.nan,
                'max_loss': float(self.merged_data['% Change_calc'].min()) if self.merged_data['% Change_calc'].notna().any() else np.nan,
                'total_profit_loss_value': float(self.merged_data['Profit_Loss_Value'].sum()) if self.merged_data['Profit_Loss_Value'].notna().any() else 0
            }
            return summary
        except Exception as e:
            st.error(f"Error in performance summary: {str(e)}")
            return {}
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Analyze performance by sector."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            sector_analysis = self.merged_data.groupby('Sector_curr').agg({
                '% Change_calc': ['mean', 'median', 'std', 'count'],
                'Profit_Loss_Value': ['sum', 'mean'],
                'Volume_curr': ['mean', 'sum']
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
            industry_analysis = self.merged_data.groupby('Industry_curr').agg({
                '% Change_calc': ['mean', 'median', 'std', 'count'],
                'Profit_Loss_Value': ['sum', 'mean'],
                'Volume_curr': ['mean']
            }).round(2)
            
            industry_analysis.columns = ['_'.join(col).strip() for col in industry_analysis.columns]
            industry_analysis = industry_analysis.reset_index()
            industry_analysis = industry_analysis.sort_values('% Change_calc_mean', ascending=False).head(20)
            return industry
