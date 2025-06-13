import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

class ComparativeAnalysis:
    def __init__(self, current_data, previous_data):
        """Initialize with current and previous stock data."""
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = self._merge_data()
    
    def _merge_data(self):
        """Merge current and previous data on Symbol."""
        try:
            if 'Symbol' not in self.current_data.columns or 'Symbol' not in self.previous_data.columns:
                return pd.DataFrame()
            
            # Clean Symbols
            self.current_data['Symbol'] = self.current_data['Symbol'].astype(str).str.strip()
            self.previous_data['Symbol'] = self.previous_data['Symbol'].astype(str).str.strip()
            
            # Merge
            merged = pd.merge(
                self.current_data,
                self.previous_data,
                on='Symbol',
                how='inner',
                suffixes=('_curr', '_prev')
            )
            
            # Calculate price change
            if 'Last Sale_curr' in merged.columns and 'Last Sale_prev' in merged.columns:
                merged['% Change_calc'] = ((merged['Last Sale_curr'] - merged['Last Sale_prev']) / merged['Last Sale_prev']) * 100
                merged['Profit/Loss'] = merged['Last Sale_curr'] - merged['Last Sale_prev']
            
            return merged
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_summary(self):
        """Return performance summary metrics."""
        if self.merged_data.empty or '% Change_calc' not in self.merged_data.columns:
            return {}
        
        valid_data = self.merged_data.dropna(subset=['% Change_calc'])
        if valid_data.empty:
            return {}
        
        return {
            'total_stocks': len(valid_data),
            'avg_change': valid_data['% Change_calc'].mean(),
            'gainers': len(valid_data[valid_data['% Change_calc'] > 0]),
            'losers': len(valid_data[valid_data['% Change_calc'] < 0])
        }
    
    def create_performance_dashboard(self):
        """Create a performance dashboard."""
        if self.merged_data.empty or '% Change_calc' not in self.merged_data.columns:
            return None
        
        valid_data = self.merged_data.dropna(subset=['% Change_calc'])
        if valid_data.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_data['% Change_calc'],
            name='Price Change (%)',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Distribution of Price Changes",
            xaxis_title="% Change",
            yaxis_title="Number of Stocks",
            height=500
        )
        return fig
    
    def get_sector_analysis(self):
        """Return sector-level performance analysis."""
        if self.merged_data.empty or 'Sector_curr' not in self.merged_data.columns:
            return pd.DataFrame()
        
        return self.merged_data.groupby('Sector_curr').agg({
            '% Change_calc': ['mean', 'count']
        }).reset_index().round(2)
    
    def get_industry_analysis(self):
        """Return industry-level performance analysis."""
        if self.merged_data.empty or 'Industry_curr' not in self.merged_data.columns:
            return pd.DataFrame()
        
        return self.merged_data.groupby('Industry_curr').agg({
            '% Change_calc': ['mean', 'count']
        }).reset_index().round(2).nlargest(20, ('% Change_calc', 'mean'))
    
    def detect_outliers(self):
        """Detect extreme gainers and losers."""
        if self.merged_data.empty or '% Change_calc' not in self.merged_data.columns:
            return {}
        
        valid_data = self.merged_data.dropna(subset=['% Change_calc'])
        if valid_data.empty:
            return {}
        
        q1 = valid_data['% Change_calc'].quantile(0.25)
        q3 = valid_data['% Change_calc'].quantile(0.75)
        iqr = q3 - q1
        extreme_gainers = valid_data[valid_data['% Change_calc'] > q3 + 1.5 * iqr][['Symbol', '% Change_calc']].to_dict('records')
        extreme_losers = valid_data[valid_data['% Change_calc'] < q1 - 1.5 * iqr][['Symbol', '% Change_calc']].to_dict('records')
        
        return {
            'extreme_gainers': extreme_gainers,
            'extreme_losers': extreme_losers
        }
    
    def calculate_correlations(self):
        """Calculate correlation matrix for key metrics."""
        if self.merged_data.empty:
            return pd.DataFrame()
        
        numeric_cols = ['Last Sale_curr', 'Last Sale_prev', '% Change_calc', 'Profit/Loss']
        valid_cols = [col for col in numeric_cols if col in self.merged_data.columns]
        if not valid_cols:
            return pd.DataFrame()
        
        return self.merged_data[valid_cols].corr().round(2)
