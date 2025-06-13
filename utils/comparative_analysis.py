"""
Comparative Analysis Module for Phase 1
Compares current and previous stock data
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List

class ComparativeAnalysis:
    """Perform comparison between current and previous stock data."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._compare_data()
    
    def _compare_data(self):
        """Compare current and previous data, calculate changes."""
        try:
            # Find symbol column
            symbol_col_current = self._find_symbol_column(self.current_data)
            symbol_col_previous = self._find_symbol_column(self.previous_data)
            
            if not symbol_col_current or not symbol_col_previous:
                st.error("Symbol column not found. Ensure 'Symbol' or 'Ticker' exists.")
                return
            
            # Standardize data
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
            
            # Merge on Symbol
            merged = pd.merge(
                current_data_clean,
                previous_data_clean,
                on='Symbol',
                how='inner',
                suffixes=('_current', '_previous')
            )
            
            if merged.empty:
                st.warning(f"No matching symbols: Current={len(current_data_clean)}, Previous={len(previous_data_clean)}")
                return
            
            # Debug: Display merged columns
            st.write("Merged Data Columns:", merged.columns.tolist())
            
            # Define non-changing columns
            non_change_cols = ['Sector', 'Industry', 'Country']
            
            # Initialize result DataFrame
            result_df = pd.DataFrame({'Symbol': merged['Symbol']})
            
            # Add non-changing columns from current data
            for col in non_change_cols:
                col_current = f'{col}_current'
                if col_current in merged.columns:
                    result_df[col] = merged[col_current]
            
            # Add Name from current data
            name_col = self._find_name_column(merged, '_current')
            if name_col:
                result_df['Name'] = merged[name_col]
            
            # Identify numeric columns to compare
            numeric_cols = []
            for col in current_data_clean.columns:
                if col not in ['Symbol'] + non_change_cols:
                    try:
                        cleaned = self._clean_numeric_column(current_data_clean[col])
                        if cleaned.dtype in ['float64', 'int64']:
                            numeric_cols.append(col)
                    except:
                        pass
            
            # Compare numeric columns
            for col in numeric_cols:
                col_current = f'{col}_current'
                col_previous = f'{col}_previous'
                
                if col_current in merged.columns and col_previous in merged.columns:
                    merged[col_current] = self._clean_numeric_column(merged[col_current])
                    merged[col_previous] = self._clean_numeric_column(merged[col_previous])
                    
                    # Add current and previous values
                    result_df[f'{col}_current'] = merged[col_current]
                    result_df[f'{col}_previous'] = merged[col_previous]
                    
                    # Calculate change
                    result_df[f'{col}_Change'] = merged[col_current] - merged[col_previous]
                    
                    # Calculate percentage change for Price
                    if 'price' in col.lower() or 'sale' in col.lower() or 'close' in col.lower():
                        result_df['Price_Change_Pct'] = ((merged[col_current] - merged[col_previous]) / 
                                                        merged[col_previous].replace(0, np.nan)) * 100
                        result_df['Price_Profit_Loss'] = result_df['Price_Change_Pct'].apply(
                            lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                        )
                    
                    # Add Profit/Loss for other numeric columns
                    result_df[f'{col}_Profit_Loss'] = result_df[f'{col}_Change'].apply(
                        lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                    )
            
            self.merged_data = result_df
            st.success(f"Compared {len(result_df)} stocks")
            
        except Exception as e:
            st.error(f"Error comparing data: {str(e)}")
    
    def _find_symbol_column(self, df: pd.DataFrame) -> str:
        """Find symbol column."""
        possible_names = ['Symbol', 'Ticker', 'Stock', 'Code']
        for col in possible_names:
            if col.lower() in [c.lower() for c in df.columns]:
                return next(c for c in df.columns if c.lower() == col.lower())
        return df.columns[0] if len(df.columns) > 0 else None
    
    def _find_name_column(self, df: pd.DataFrame, suffix: str) -> str:
        """Find name column."""
        possible_names = [f'Name{suffix}', f'Company{suffix}']
        for col in possible_names:
            if col in df.columns:
                return col
        return None
    
    def _find_price_column(self, df: pd.DataFrame, suffix: str) -> str:
        """Find price column."""
        possible_names = [
            f'Last Sale{suffix}', f'Price{suffix}', f'Close{suffix}', 
            f'Last{suffix}', f'Closing Price{suffix}', f'Current{suffix}'
        ]
        for col in possible_names:
            if col in df.columns:
                return col
        
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['price', 'sale', 'close', 'last', 'current', 'closing']) and suffix.lower() in col_lower:
                return col
        
        for col in df.columns:
            if suffix.lower() in col.lower():
                try:
                    cleaned = self._clean_numeric_column(df[col])
                    if cleaned.dtype in ['float64', 'int64']:
                        st.warning(f"No standard price column, using: {col}")
                        return col
                except:
                    pass
        return None
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert to numeric."""
        if series.dtype in ['object', 'string']:
            cleaned = series.astype(str).str.strip()
            cleaned = cleaned.str.replace(r'[$,€£¥₹]', '', regex=True)
            cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
            cleaned = cleaned.replace(['', 'N/A', 'nan', 'NaN'], np.nan)
            cleaned = pd.to_numeric(cleaned, errors='coerce')
        else:
            cleaned = pd.to_numeric(series, errors='coerce')
        return cleaned
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        summary = {}
        
        try:
            if 'Price_Change_Pct' in self.merged_data.columns:
                price_changes = self.merged_data['Price_Change_Pct'].dropna()
                if not price_changes.empty:
                    summary.update({
                        'total_stocks': len(self.merged_data),
                        'gainers': int((price_changes > 0).sum()),
                        'losers': int((price_changes < 0).sum()),
                        'unchanged': int((price_changes == 0).sum()),
                        'avg_change': float(price_changes.mean()),
                        'max_gain': float(price_changes.max()),
                        'max_loss': float(price_changes.min())
                    })
            return summary
        
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return {
                'total_stocks': 0,
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change': 0.0,
                'max_gain': 0.0,
                'max_loss': 0.0
            }
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create performance dashboard."""
        if self.merged_data is None or self.merged_data.empty or 'Price_Change_Pct' not in self.merged_data.columns:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "histogram"}, {"type": "pie"}]],
            subplot_titles=("Price Change Distribution", "Gainers vs Losers")
        )
        
        price_changes = self.merged_data['Price_Change_Pct'].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=price_changes,
                nbinsx=30,
                name="Price Changes",
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        gainers = (price_changes > 0).sum()
        losers = (price_changes < 0).sum()
        unchanged = (price_changes == 0).sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Gainers', 'Losers', 'Unchanged'],
                values=[gainers, losers, unchanged],
                name="Performance",
                marker_colors=['#2ca02c', '#d62728', '#7f7f7f']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Stock Performance Dashboard",
            showlegend=True,
            height=400
        )
        
        return fig
