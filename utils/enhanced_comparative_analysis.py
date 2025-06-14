"""
Enhanced Comparative Analysis with All Restored Features
- Stock symbol search
- Sector filtering
- Interactive charts with hover
- Per-stock statistical analysis
- Sector/Industry/Correlation analysis
- Performance dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from datetime import datetime, timedelta

class EnhancedComparativeAnalysis:
    """Complete comparative analysis with all original features restored."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and merge datasets with accurate calculations."""
        try:
            # Find symbol columns
            current_symbol = self._find_symbol_column(self.current_data)
            previous_symbol = self._find_symbol_column(self.previous_data)
            
            if not current_symbol or not previous_symbol:
                st.error("Symbol column not found in datasets")
                return
            
            # Clean datasets
            current_clean = self.current_data.copy()
            previous_clean = self.previous_data.copy()
            
            # Standardize symbol columns
            current_clean['Symbol'] = current_clean[current_symbol].astype(str).str.strip().str.upper()
            previous_clean['Symbol'] = previous_clean[previous_symbol].astype(str).str.strip().str.upper()
            
            # Remove invalid symbols
            current_clean = current_clean[current_clean['Symbol'].notna() & (current_clean['Symbol'] != '') & (current_clean['Symbol'] != 'NAN')]
            previous_clean = previous_clean[previous_clean['Symbol'].notna() & (previous_clean['Symbol'] != '') & (previous_clean['Symbol'] != 'NAN')]
            
            # Merge datasets
            merged = pd.merge(
                current_clean, 
                previous_clean, 
                on='Symbol', 
                how='inner',
                suffixes=('_current', '_previous')
            )
            
            if merged.empty:
                st.warning("No matching symbols found between datasets")
                return
            
            # Calculate accurate price changes
            price_current_col = self._find_price_column(merged, '_current')
            price_previous_col = self._find_price_column(merged, '_previous')
            
            if price_current_col and price_previous_col:
                # Clean price data
                merged[price_current_col] = pd.to_numeric(
                    merged[price_current_col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                )
                merged[price_previous_col] = pd.to_numeric(
                    merged[price_previous_col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                )
                
                # Calculate changes with correct percentage formula
                merged['Price_Change'] = merged[price_current_col] - merged[price_previous_col]
                
                # Calculate percentage change: (Current - Previous) / Previous * 100
                merged['Price_Change_Pct'] = (
                    merged['Price_Change'] / merged[price_previous_col].replace(0, np.nan) * 100
                ).round(2)
                
                # Add standardized column names for easier access
                merged['Current_Price'] = merged[price_current_col]
                merged['Previous_Price'] = merged[price_previous_col]
            
            self.merged_data = merged
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
    
    def _find_symbol_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find symbol column with flexible naming."""
        candidates = ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'Stock']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _find_price_column(self, df: pd.DataFrame, suffix: str) -> Optional[str]:
        """Find price column with given suffix."""
        candidates = [f'Last Sale{suffix}', f'Price{suffix}', f'Close{suffix}', f'Last Price{suffix}']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def search_stock(self, symbol: str) -> pd.DataFrame:
        """Search for specific stock symbol in the data."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        symbol = symbol.upper().strip()
        result = self.merged_data[self.merged_data['Symbol'].str.contains(symbol, case=False, na=False)]
        return result
    
    def filter_by_sector(self, sector: str) -> pd.DataFrame:
        """Filter data by sector."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
        if not sector_cols:
            return pd.DataFrame()
        
        sector_col = sector_cols[0]
        if sector == "All":
            return self.merged_data
        
        return self.merged_data[self.merged_data[sector_col] == sector]
    
    def get_available_sectors(self) -> List[str]:
        """Get list of available sectors."""
        if self.merged_data is None or self.merged_data.empty:
            return []
        
        sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
        if not sector_cols:
            return []
        
        sector_col = sector_cols[0]
        sectors = self.merged_data[sector_col].dropna().unique().tolist()
        return ['All'] + sorted(sectors)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if self.merged_data is None or self.merged_data.empty or 'Price_Change_Pct' not in self.merged_data.columns:
            return {}
        
        valid_data = self.merged_data['Price_Change_Pct'].notna()
        changes = self.merged_data.loc[valid_data, 'Price_Change_Pct']
        
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
    
    def create_interactive_performance_chart(self) -> go.Figure:
        """Create interactive performance chart with hover data."""
        if self.merged_data is None or self.merged_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Scatter plot with hover information
        hover_text = []
        for idx, row in self.merged_data.iterrows():
            hover_info = f"<b>{row.get('Symbol', 'N/A')}</b><br>"
            hover_info += f"Current Price: ${row.get('Current_Price', 0):.2f}<br>"
            hover_info += f"Previous Price: ${row.get('Previous_Price', 0):.2f}<br>"
            hover_info += f"Price Change: ${row.get('Price_Change', 0):.2f}<br>"
            hover_info += f"% Change: {row.get('Price_Change_Pct', 0):.2f}%<br>"
            
            # Add sector info if available
            sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
            if sector_cols:
                hover_info += f"Sector: {row.get(sector_cols[0], 'N/A')}"
            
            hover_text.append(hover_info)
        
        fig.add_trace(go.Scatter(
            x=self.merged_data['Previous_Price'],
            y=self.merged_data['Current_Price'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.merged_data['Price_Change_Pct'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="% Change")
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Stocks'
        ))
        
        # Add diagonal line for reference
        max_price = max(self.merged_data['Previous_Price'].max(), self.merged_data['Current_Price'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_price],
            y=[0, max_price],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='No Change Line',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Stock Performance: Current vs Previous Prices",
            xaxis_title="Previous Price ($)",
            yaxis_title="Current Price ($)",
            hovermode='closest'
        )
        
        return fig
    
    def create_sector_performance_chart(self) -> go.Figure:
        """Create interactive sector performance chart."""
        if self.merged_data is None or self.merged_data.empty:
            return go.Figure()
        
        sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
        if not sector_cols:
            return go.Figure()
        
        sector_col = sector_cols[0]
        sector_performance = self.merged_data.groupby(sector_col)['Price_Change_Pct'].agg(['mean', 'count']).reset_index()
        sector_performance = sector_performance[sector_performance['count'] >= 2]  # At least 2 stocks
        
        fig = go.Figure(data=[
            go.Bar(
                x=sector_performance[sector_col],
                y=sector_performance['mean'],
                text=[f"{x:.1f}%" for x in sector_performance['mean']],
                textposition='auto',
                marker_color=['green' if x > 0 else 'red' for x in sector_performance['mean']],
                hovertemplate='<b>%{x}</b><br>Average Change: %{y:.2f}%<br>Stock Count: %{customdata}<extra></extra>',
                customdata=sector_performance['count']
            )
        ])
        
        fig.update_layout(
            title="Average Performance by Sector",
            xaxis_title="Sector",
            yaxis_title="Average % Change",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Get detailed sector analysis."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
        if not sector_cols:
            return pd.DataFrame()
        
        sector_col = sector_cols[0]
        analysis = self.merged_data.groupby(sector_col).agg({
            'Price_Change_Pct': ['count', 'mean', 'std', 'min', 'max'],
            'Price_Change': 'sum'
        }).round(2)
        
        # Flatten column names
        analysis.columns = ['Stock_Count', 'Avg_Change_Pct', 'Std_Dev', 'Min_Change', 'Max_Change', 'Total_Dollar_Change']
        analysis = analysis.reset_index()
        analysis = analysis.sort_values('Avg_Change_Pct', ascending=False)
        
        return analysis
    
    def get_industry_analysis(self) -> pd.DataFrame:
        """Get detailed industry analysis."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        industry_cols = [col for col in self.merged_data.columns if 'industry' in col.lower()]
        if not industry_cols:
            return pd.DataFrame()
        
        industry_col = industry_cols[0]
        analysis = self.merged_data.groupby(industry_col).agg({
            'Price_Change_Pct': ['count', 'mean', 'std', 'min', 'max'],
            'Price_Change': 'sum'
        }).round(2)
        
        # Flatten column names
        analysis.columns = ['Stock_Count', 'Avg_Change_Pct', 'Std_Dev', 'Min_Change', 'Max_Change', 'Total_Dollar_Change']
        analysis = analysis.reset_index()
        analysis = analysis.sort_values('Avg_Change_Pct', ascending=False)
        
        return analysis.head(20)  # Top 20 industries
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for numerical metrics."""
        if self.merged_data is None or self.merged_data.empty:
            return pd.DataFrame()
        
        # Select numerical columns for correlation
        numerical_cols = []
        for col in self.merged_data.columns:
            if self.merged_data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
        
        if len(numerical_cols) < 2:
            return pd.DataFrame()
        
        correlation_data = self.merged_data[numerical_cols].corr()
        return correlation_data
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create interactive correlation heatmap."""
        correlation_matrix = self.calculate_correlations()
        
        if correlation_matrix.empty:
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Correlation Matrix of Financial Metrics",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def get_stock_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific stock."""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        stock_data = self.merged_data[self.merged_data['Symbol'] == symbol.upper()]
        if stock_data.empty:
            return {}
        
        stock = stock_data.iloc[0]
        
        stats = {
            'symbol': stock.get('Symbol', 'N/A'),
            'current_price': stock.get('Current_Price', 0),
            'previous_price': stock.get('Previous_Price', 0),
            'price_change': stock.get('Price_Change', 0),
            'price_change_pct': stock.get('Price_Change_Pct', 0)
        }
        
        # Add sector/industry if available
        sector_cols = [col for col in self.merged_data.columns if 'sector' in col.lower()]
        if sector_cols:
            stats['sector'] = stock.get(sector_cols[0], 'N/A')
        
        industry_cols = [col for col in self.merged_data.columns if 'industry' in col.lower()]
        if industry_cols:
            stats['industry'] = stock.get(industry_cols[0], 'N/A')
        
        return stats
    
    def create_performance_dashboard(self) -> go.Figure:
        """Create comprehensive performance dashboard."""
        if self.merged_data is None or self.merged_data.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price Distribution', 'Change Distribution', 'Top Performers', 'Bottom Performers'],
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=self.merged_data['Current_Price'], name='Current Prices', nbinsx=20),
            row=1, col=1
        )
        
        # Change distribution
        fig.add_trace(
            go.Histogram(x=self.merged_data['Price_Change_Pct'], name='% Changes', nbinsx=20),
            row=1, col=2
        )
        
        # Top performers
        top_5 = self.merged_data.nlargest(5, 'Price_Change_Pct')
        fig.add_trace(
            go.Bar(x=top_5['Symbol'], y=top_5['Price_Change_Pct'], name='Top 5', marker_color='green'),
            row=2, col=1
        )
        
        # Bottom performers
        bottom_5 = self.merged_data.nsmallest(5, 'Price_Change_Pct')
        fig.add_trace(
            go.Bar(x=bottom_5['Symbol'], y=bottom_5['Price_Change_Pct'], name='Bottom 5', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Comprehensive Performance Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig