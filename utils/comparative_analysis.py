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
    
    def clean_numeric(self, value):
        """Handle currency symbols, commas, etc., to clean and convert them to numeric"""
        try:
            return float(str(value).replace('$', '').replace(',', '').replace('%', '').strip())
        except ValueError:
            return 0  # Return 0 if it's not a valid number

    def _prepare_comparative_data(self):
        """Prepare merged dataset for comparative analysis using logic from COMPARE_PRE_CURR_EAR_PRICE.py."""
        try:
            # Find symbol column
            symbol_col_current = self._find_symbol_column(self.current_data)
            symbol_col_previous = self._find_symbol_column(self.previous_data)
            
            if not symbol_col_current or not symbol_col_previous:
                st.error(f"Symbol column not found. Please ensure your data has a column named Symbol, Ticker, or similar")
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
            
            # Validate required columns
            required_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', 'Industry']
            if not all(col in current_data_clean.columns for col in required_columns) or \
               not all(col in previous_data_clean.columns for col in required_columns):
                st.error("The input data must contain the required columns: 'Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', and 'Industry'")
                return
            
            # Merge datasets on Symbol
            merged = pd.merge(
                current_data_clean, 
                previous_data_clean, 
                on='Symbol', 
                how='inner',
                suffixes=('_curr', '_prev')
            )
            
            if merged.empty:
                st.warning(f"No matching symbols found. Current: {len(current_data_clean)}, Previous: {len(previous_data_clean)}")
                return
            
            # Clean and convert necessary columns to numeric
            for col in ['Last Sale', 'Net Change', '% Change']:
                merged[f'{col}_prev'] = merged[f'{col}_prev'].apply(self.clean_numeric)
                merged[f'{col}_curr'] = merged[f'{col}_curr'].apply(self.clean_numeric)
            
            # Calculate Profit/Loss and % Change
            merged['Profit/Loss'] = merged['Last Sale_curr'] - merged['Last Sale_prev']
            merged['% Change_calc'] = ((merged['Profit/Loss'] / merged['Last Sale_prev'].replace(0, np.nan)) * 100).round(2)
            
            # Add Profit/Loss classification
            merged['Profit_Loss'] = merged['% Change_calc'].apply(
                lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
            )
            
            # Process volume columns
            volume_col_current = self._find_volume_column(merged, '_curr')
            volume_col_previous = self._find_volume_column(merged, '_prev')
            
            if volume_col_current and volume_col_previous:
                merged[volume_col_current] = self._clean_numeric_column(merged[volume_col_current])
                merged[volume_col_previous] = self._clean_numeric_column(merged[volume_col_previous])
                
                merged['Volume_Change'] = merged[volume_col_current] - merged[volume_col_previous]
                merged['Volume_Change_Pct'] = ((merged[volume_col_current] - merged[volume_col_previous]) / 
                                             merged[volume_col_previous].replace(0, np.nan)) * 100
            
            # Process market cap columns
            mcap_col_current = self._find_market_cap_column(merged, '_curr')
            mcap_col_previous = self._find_market_cap_column(merged, '_prev')
            
            if mcap_col_current and mcap_col_previous:
                merged[mcap_col_current] = self._clean_numeric_column(merged[mcap_col_current])
                merged[mcap_col_previous] = self._clean_numeric_column(merged[mcap_col_previous])
                
                merged['MarketCap_Change'] = merged[mcap_col_current] - merged[mcap_col_previous]
                merged['MarketCap_Change_Pct'] = ((merged[mcap_col_current] - merged[mcap_col_previous]) / 
                                                merged[mcap_col_previous].replace(0, np.nan)) * 100
            
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
        
        return cleaned
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            if '% Change_calc' in self.merged_data.columns:
                price_changes = self.merged_data['% Change_calc'].dropna()
                if not price_changes.empty:
                    gainers = int((price_changes > 0).sum())
                    losers = int((price_changes < 0).sum())
                    unchanged = int((price_changes == 0).sum())
                    avg_change = float(price_changes.mean())
                    
                    summary = {
                        'total_stocks': len(self.merged_data),
                        'gainers': gainers,
                        'losers': losers,
                        'unchanged': unchanged,
                        'avg_change': avg_change,
                        'max_gain': float(price_changes.max()),
                        'max_loss': float(price_changes.min())
                    }
                    
                    return summary
            
            return {
                'total_stocks': len(self.merged_data),
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change': 0.0,
                'max_gain': 0.0,
                'max_loss': 0.0
            }
            
        except Exception as e:
            st.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Analyze performance by sector."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            sector_col = 'Sector_curr' if 'Sector_curr' in self.merged_data.columns else None
            if not sector_col:
                return pd.DataFrame()
            
            sector_analysis = self.merged_data.groupby(sector_col).agg({
                '% Change_calc': ['mean', 'median', 'std', 'count'],
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
            industry_col = 'Industry_curr' if 'Industry_curr' in self.merged_data.columns else None
            if not industry_col:
                return pd.DataFrame()
            
            industry_analysis = self.merged_data.groupby(industry_col).agg({
                '% Change_calc': ['mean', 'median', 'std', 'count'],
                'MarketCap_Change_Pct': ['mean'] if 'MarketCap_Change_Pct' in self.merged_data.columns else ['count'],
                'Volume_Change_Pct': ['mean'] if 'Volume_Change_Pct' in self.merged_data.columns else ['count']
            }).round(2)
            
            industry_analysis.columns = ['_'.join(col).strip() for col in industry_analysis.columns]
            industry_analysis = industry_analysis.reset_index()
            
            if '% Change_calc_mean' in industry_analysis.columns:
                industry_analysis = industry_analysis.sort_values('% Change_calc_mean', ascending=False).head(20)
            
            return industry_analysis
            
        except Exception as e:
            st.error(f"Error in industry analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_country_analysis(self) -> pd.DataFrame:
        """Analyze performance by country."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            country_col = 'Country_curr' if 'Country_curr' in self.merged_data.columns else None
            if not country_col:
                return pd.DataFrame()
            
            country_analysis = self.merged_data.groupby(country_col).agg({
                '% Change_calc': ['mean', 'median', 'count'],
                'Market Cap_curr': ['sum', 'mean'] if 'Market Cap_curr' in self.merged_data.columns else ['count'],
                'Volume_curr': ['sum', 'mean'] if 'Volume_curr' in self.merged_data.columns else ['count']
            }).round(2)
            
            country_analysis.columns = ['_'.join(col).strip() for col in country_analysis.columns]
            country_analysis = country_analysis.reset_index()
            
            return country_analysis
            
        except Exception as e:
            st.error(f"Error in country analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_ipo_year_analysis(self) -> pd.DataFrame:
        """Analyze performance by IPO year."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            ipo_col = 'IPO Year_curr' if 'IPO Year_curr' in self.merged_data.columns else None
            if not ipo_col:
                return pd.DataFrame()
            
            self.merged_data['IPO_Decade'] = (self.merged_data[ipo_col] // 10) * 10
            
            ipo_analysis = self.merged_data.groupby('IPO_Decade').agg({
                '% Change_calc': ['mean', 'median', 'count'],
                'Market Cap_curr': ['mean'] if 'Market Cap_curr' in self.merged_data.columns else ['count']
            }).round(2)
            
            ipo_analysis.columns = ['_'.join(col).strip() for col in ipo_analysis.columns]
            ipo_analysis = ipo_analysis.reset_index()
            
            return ipo_analysis
            
        except Exception as e:
            st.error(f"Error in IPO year analysis: {str(e)}")
            return pd.DataFrame()
    
    def detect_outliers(self) -> Dict:
        """Detect outlier stocks with extreme changes."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            outliers = {}
            
            if '% Change_calc' in self.merged_data.columns:
                price_changes = self.merged_data['% Change_calc'].dropna()
                mean_change = price_changes.mean()
                std_change = price_changes.std()
                
                extreme_threshold = 3
                extreme_positive = self.merged_data[
                    self.merged_data['% Change_calc'] > (mean_change + extreme_threshold * std_change)
                ][['Symbol', 'Name_curr', '% Change_calc', 'Last Sale_curr']].to_dict('records')
                
                extreme_negative = self.merged_data[
                    self.merged_data['% Change_calc'] < (mean_change - extreme_threshold * std_change)
                ][['Symbol', 'Name_curr', '% Change_calc', 'Last Sale_curr']].to_dict('records')
                
                outliers['extreme_gainers'] = extreme_positive
                outliers['extreme_losers'] = extreme_negative
            
            if 'Volume_Change_Pct' in self.merged_data.columns:
                volume_changes = self.merged_data['Volume_Change_Pct'].dropna()
                if not volume_changes.empty:
                    vol_mean = volume_changes.mean()
                    vol_std = volume_changes.std()
                    
                    high_volume_activity = self.merged_data[
                        self.merged_data['Volume_Change_Pct'] > (vol_mean + 2 * vol_std)
                    ][['Symbol', 'Name_curr', 'Volume_Change_Pct', 'Volume_curr']].to_dict('records')
                    
                    outliers['high_volume_activity'] = high_volume_activity
            
            return outliers
            
        except Exception as e:
            st.error(f"Error detecting outliers: {str(e)}")
            return {}
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for numerical metrics."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        try:
            numerical_cols = [col for col in self.merged_data.columns if 
                            self.merged_data[col].dtype in ['float64', 'int64'] and 
                            ('Change' in col or 'Profit/Loss' in col)]
            
            if len(numerical_cols) < 2:
                return pd.DataFrame()
            
            correlation_matrix = self.merged_data[numerical_cols].corr()
            return correlation_matrix
            
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard."""
        if self.merged_data is None or self.merged_data.empty:
            return go.Figure()
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Change Distribution', 'Sector Performance', 
                              'Price vs Market Cap Change', 'Volume Change Analysis'),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "box"}]]
            )
            
            if '% Change_calc' in self.merged_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=self.merged_data['% Change_calc'],
                        name='Price Change %',
                        nbinsx=30,
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
            
            if 'Sector_curr' in self.merged_data.columns and '% Change_calc' in self.merged_data.columns:
                sector_perf = self.merged_data.groupby('Sector_curr')['% Change_calc'].mean().sort_values(ascending=False)
                fig.add_trace(
                    go.Bar(
                        x=sector_perf.index,
                        y=sector_perf.values,
                        name='Avg Sector Performance',
                        marker_color='lightgreen'
                    ),
                    row=1, col=2
                )
            
            if '% Change_calc' in self.merged_data.columns and 'MarketCap_Change_Pct' in self.merged_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.merged_data['% Change_calc'],
                        y=self.merged_data['MarketCap_Change_Pct'],
                        mode='markers',
                        name='Price vs Market Cap Change',
                        text=self.merged_data['Symbol'],
                        marker=dict(size=8, color='coral')
                    ),
                    row=2, col=1
                )
            
            if 'Volume_Change_Pct' in self.merged_data.columns:
                fig.add_trace(
                    go.Box(
                        y=self.merged_data['Volume_Change_Pct'],
                        name='Volume Change %',
                        marker_color='lightcoral'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title="Comprehensive Performance Dashboard",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")
            return go.Figure()
