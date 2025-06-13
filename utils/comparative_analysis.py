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
            # Validate required columns
            required_columns = ['Symbol', 'Last Sale']
            if not all(col in self.current_data.columns for col in required_columns) or \
               not all(col in self.previous_data.columns for col in required_columns):
                st.error(f"Input data must contain {required_columns} columns")
                return
            
            # Debug: Log input DataFrame columns
            st.write(f"Current data columns: {list(self.current_data.columns)}")
            st.write(f"Previous data columns: {list(self.previous_data.columns)}")
            
            # Debug: Log raw Last Sale values
            st.write(f"Raw Last Sale_current sample: {self.current_data['Last Sale'].head().tolist()}")
            st.write(f"Raw Last Sale_previous sample: {self.previous_data['Last Sale'].head().tolist()}")
            
            # Standardize symbol column names
            current_data_clean = self.current_data.copy()
            previous_data_clean = self.previous_data.copy()
            
            # Clean and standardize symbols
            current_data_clean['Symbol'] = current_data_clean['Symbol'].astype(str).str.strip().str.upper()
            previous_data_clean['Symbol'] = previous_data_clean['Symbol'].astype(str).str.strip().str.upper()
            
            # Remove empty symbols
            current_data_clean = current_data_clean[current_data_clean['Symbol'].notna() & (current_data_clean['Symbol'] != '')]
            previous_data_clean = previous_data_clean[previous_data_clean['Symbol'].notna() & (previous_data_clean['Symbol'] != '')]
            
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
            
            # Debug: Log merged DataFrame columns
            st.write(f"Merged data columns: {list(merged.columns)}")
            
            # Clean Last Sale columns
            merged['Last Sale_current'] = self._clean_numeric_column(merged['Last Sale_current'])
            merged['Last Sale_previous'] = self._clean_numeric_column(merged['Last Sale_previous'])
            
            # Debug: Log cleaned Last Sale values and check for NaN/None
            st.write(f"Cleaned Last Sale_current sample: {merged['Last Sale_current'].head().tolist()}")
            st.write(f"Cleaned Last Sale_previous sample: {merged['Last Sale_previous'].head().tolist()}")
            st.write(f"NaN in Last Sale_current: {merged['Last Sale_current'].isna().sum()}")
            st.write(f"NaN in Last Sale_previous: {merged['Last Sale_previous'].isna().sum()}")
            
            # Calculate price changes
            merged['Price_Change'] = merged['Last Sale_current'] - merged['Last Sale_previous']
            merged['Price_Change_Pct'] = ((merged['Price_Change'] / merged['Last Sale_previous'].replace(0, np.nan)) * 100).round(2)
            
            # Add profit/loss classification
            merged['Profit_Loss'] = merged['Price_Change_Pct'].apply(
                lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
            )
            
            # Clean optional numeric columns if present
            for col in ['Net Change', '% Change', 'Volume', 'Market Cap']:
                if f'{col}_current' in merged.columns:
                    merged[f'{col}_current'] = self._clean_numeric_column(merged[f'{col}_current'])
                if f'{col}_previous' in merged.columns:
                    merged[f'{col}_previous'] = self._clean_numeric_column(merged[f'{col}_previous'])
            
            # Calculate volume and market cap changes if columns exist
            if 'Volume_current' in merged.columns and 'Volume_previous' in merged.columns:
                merged['Volume_Change'] = merged['Volume_current'] - merged['Volume_previous']
                merged['Volume_Change_Pct'] = ((merged['Volume_current'] - merged['Volume_previous']) / 
                                              merged['Volume_previous'].replace(0, np.nan)) * 100
            
            if 'Market Cap_current' in merged.columns and 'Market Cap_previous' in merged.columns:
                merged['MarketCap_Change'] = merged['Market Cap_current'] - merged['Market Cap_previous']
                merged['MarketCap_Change_Pct'] = ((merged['Market Cap_current'] - merged['Market Cap_previous']) / 
                                                 merged['Market Cap_previous'].replace(0, np.nan)) * 100
            
            self.merged_data = merged
            st.success(f"Successfully merged {len(merged)} stocks for comparative analysis")
            
        except Exception as e:
            st.error(f"Error preparing comparative data: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric values, handling currency symbols, commas, and percentages."""
        try:
            # Debug: Log raw sample values
            st.write(f"Cleaning column: {series.name}, Raw sample values: {series.head().tolist()}")
            
            # Convert to string, handle None and non-string types
            cleaned = series.astype(str).str.strip()
            
            # Replace common non-numeric characters
            cleaned = cleaned.str.replace('$', '', regex=False)
            cleaned = cleaned.str.replace(',', '', regex=False)
            cleaned = cleaned.str.replace('€', '', regex=False)
            cleaned = cleaned.str.replace('£', '', regex=False)
            cleaned = cleaned.str.replace('¥', '', regex=False)
            cleaned = cleaned.str.replace('₹', '', regex=False)
            cleaned = cleaned.str.replace('%', '', regex=False)
            
            # Replace empty strings or invalid entries
            cleaned = cleaned.replace('', '0').replace('None', '0').replace('nan', '0')
            
            # Log invalid values
            invalid = cleaned[~cleaned.str.replace('-', '').str.replace('.', '').str.isnumeric()]
            if not invalid.empty:
                st.warning(f"Invalid values in {series.name}: {invalid.head().tolist()}")
            
            # Convert to numeric
            cleaned = pd.to_numeric(cleaned, errors='coerce').fillna(0)
            
            # Debug: Log cleaned sample values
            st.write(f"Cleaned sample values for {series.name}: {cleaned.head().tolist()}")
            
            return cleaned
        except Exception as e:
            st.error(f"Error cleaning numeric column: {str(e)}")
            return pd.to_numeric(series.astype(str), errors='coerce').fillna(0)
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        try:
            if 'Price_Change_Pct' in self.merged_data.columns:
                price_changes = self.merged_data['Price_Change_Pct'].dropna()
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
            return {
                'total_stocks': 0,
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change': 0.0,
                'max_gain': 0.0,
                'max_loss': 0.0
            }
    
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
                return pd.DataFrame()
            
            country_analysis = self.merged_data.groupby(country_col).agg({
                'Price_Change_Pct': ['mean', 'median', 'count'],
                'Market Cap_current': ['sum', 'mean'] if 'Market Cap_current' in self.merged_data.columns else ['count'],
                'Volume_current': ['sum', 'mean'] if 'Volume_current' in self.merged_data.columns else ['count']
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
            ipo_col = 'IPO Year_current' if 'IPO Year_current' in self.merged_data.columns else None
            if not ipo_col:
                return pd.DataFrame()
            
            self.merged_data['IPO_Decade'] = (self.merged_data[ipo_col] // 10) * 10
            
            ipo_analysis = self.merged_data.groupby('IPO_Decade').agg({
                'Price_Change_Pct': ['mean', 'median', 'count'],
                'Market Cap_current': ['mean'] if 'Market Cap_current' in self.merged_data.columns else ['count']
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
            
            if 'Price_Change_Pct' in self.merged_data.columns:
                # Debug: Log Last Sale_current values
                st.write(f"Last Sale_current in outliers: {self.merged_data['Last Sale_current'].dropna().head().tolist()}")
                
                price_changes = self.merged_data['Price_Change_Pct'].dropna()
                mean_change = price_changes.mean()
                std_change = price_changes.std()
                
                extreme_threshold = 3
                # Ensure NaN is replaced with 0 for Last Sale_current
                outlier_data = self.merged_data.copy()
                outlier_data['Last Sale_current'] = outlier_data['Last Sale_current'].fillna(0)
                
                extreme_positive = outlier_data[
                    outlier_data['Price_Change_Pct'] > (mean_change + extreme_threshold * std_change)
                ][['Symbol', 'Name_current', 'Price_Change_Pct', 'Last Sale_current']].to_dict('records')
                
                extreme_negative = outlier_data[
                    outlier_data['Price_Change_Pct'] < (mean_change - extreme_threshold * std_change)
                ][['Symbol', 'Name_current', 'Price_Change_Pct', 'Last Sale_current']].to_dict('records')
                
                outliers['extreme_gainers'] = extreme_positive
                outliers['extreme_losers'] = extreme_negative
            
            if 'Volume_Change_Pct' in self.merged_data.columns:
                volume_changes = self.merged_data['Volume_Change_Pct'].dropna()
                if not volume_changes.empty:
                    vol_mean = volume_changes.mean()
                    vol_std = volume_changes.std()
                    
                    high_volume_activity = self.merged_data[
                        self.merged_data['Volume_Change_Pct'] > (vol_mean + 2 * vol_std)
                    ][['Symbol', 'Name_current', 'Volume_Change_Pct', 'Volume_current']].to_dict('records')
                    
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
                            'Change' in col]
            
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
            
            if 'Price_Change_Pct' in self.merged_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=self.merged_data['Price_Change_Pct'],
                        name='Price Change %',
                        nbinsx=30,
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
            
            if 'Sector_current' in self.merged_data.columns and 'Price_Change_Pct' in self.merged_data.columns:
                sector_perf = self.merged_data.groupby('Sector_current')['Price_Change_Pct'].mean().sort_values(ascending=False)
                fig.add_trace(
                    go.Bar(
                        x=sector_perf.index,
                        y=sector_perf.values,
                        name='Avg Sector Performance',
                        marker_color='lightgreen'
                    ),
                    row=1, col=2
                )
            
            if 'Price_Change_Pct' in self.merged_data.columns and 'MarketCap_Change_Pct' in self.merged_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.merged_data['Price_Change_Pct'],
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
