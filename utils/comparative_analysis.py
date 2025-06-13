import pandas as pd
import numpy as np
import streamlit as st

class ComparativeAnalysis:
    """Perform comparative analysis between current and previous stock data."""
    
    def __init__(self, current_data: pd.DataFrame, previous_data: pd.DataFrame):
        self.current_data = current_data.copy()
        self.previous_data = previous_data.copy()
        self.merged_data = None
        self._prepare_comparative_data()
    
    def _prepare_comparative_data(self):
        """Prepare merged dataset for comparative analysis."""
        try:
            current_data_clean = self.current_data.copy()
            previous_data_clean = self.previous_data.copy()
            
            # Standardize Symbol column
            current_data_clean['Symbol'] = current_data_clean['Symbol'].astype(str).str.strip().str.upper()
            previous_data_clean['Symbol'] = previous_data_clean['Symbol'].astype(str).str.strip().str.upper()
            
            current_data_clean = current_data_clean[current_data_clean['Symbol'].notna() & (current_data_clean['Symbol'] != '')]
            previous_data_clean = previous_data_clean[previous_data_clean['Symbol'].notna() & (previous_data_clean['Symbol'] != '')]
            
            current_data_clean = current_data_clean.drop_duplicates(subset=['Symbol'], keep='first')
            previous_data_clean = previous_data_clean.drop_duplicates(subset=['Symbol'], keep='first')
            
            st.info(f"Current data: {len(current_data_clean)} rows, Previous data: {len(previous_data_clean)} rows")
            
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
            
            st.info(f"Merged data: {len(merged)} rows")
            
            # Calculate Price_Change_Pct
            price_col_current = None
            price_col_previous = None
            for col in merged.columns:
                if 'Last Sale_current' in col or 'Price_current' in col:
                    price_col_current = col
                if 'Last Sale_previous' in col or 'Price_previous' in col:
                    price_col_previous = col
            
            if price_col_current and price_col_previous:
                merged[price_col_current] = self._clean_numeric_column(merged[price_col_current])
                merged[price_col_previous] = self._clean_numeric_column(merged[price_col_previous])
                
                merged['Price_Change_Pct'] = (
                    (merged[price_col_current] - merged[price_col_previous]) /
                    merged[price_col_previous].replace(0, np.nan)
                ) * 100
                
                merged['Profit_Loss'] = merged['Price_Change_Pct'].apply(
                    lambda x: 'Profit' if x > 0 else ('Loss' if x < 0 else 'Neutral') if pd.notna(x) else 'Unknown'
                )
            else:
                st.warning(f"No valid price columns found. Available columns: {list(merged.columns)}")
                return
            
            initial_count = len(merged)
            merged = merged.dropna(subset=['Price_Change_Pct'])
            if len(merged) < initial_count:
                st.warning(f"Removed {initial_count - len(merged)} rows with invalid Price_Change_Pct values")
            
            self.merged_data = merged
            st.success(f"Successfully merged {len(merged)} stocks for comparative analysis")
        
        except Exception as e:
            st.error(f"Error preparing comparative data: {str(e)}")
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean and convert column to numeric values."""
        try:
            if series.dtype in ['object', 'string']:
                cleaned = series.astype(str).str.strip()
                cleaned = cleaned.str.replace(r'[%,$,€,£,¥,₹]', '', regex=True)
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
    
    def detect_outliers(self) -> dict:
        """Detect extreme gainers and losers using IQR method."""
        if self.merged_data is None or self.merged_data.empty or 'Price_Change_Pct' not in self.merged_data.columns:
            return {'extreme_gainers': [], 'extreme_losers': []}
        
        try:
            valid_data = self.merged_data[self.merged_data['Price_Change_Pct'].notna()].copy()
            if valid_data.empty:
                return {'extreme_gainers': [], 'extreme_losers': []}
            
            Q1 = valid_data['Price_Change_Pct'].quantile(0.25)
            Q3 = valid_data['Price_Change_Pct'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            extreme_gainers = valid_data[valid_data['Price_Change_Pct'] > upper_bound][['Symbol', 'Price_Change_Pct']].to_dict('records')
            extreme_losers = valid_data[valid_data['Price_Change_Pct'] < lower_bound][['Symbol', 'Price_Change_Pct']].to_dict('records')
            
            st.info(f"Outlier bounds: Lower={lower_bound:.2f}%, Upper={upper_bound:.2f}%")
            
            return {
                'extreme_gainers': extreme_gainers,
                'extreme_losers': extreme_losers
            }
        except Exception as e:
            st.error(f"Error detecting outliers: {str(e)}")
            return {'extreme_gainers': [], 'extreme_losers': []}
