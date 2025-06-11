import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional

class DataProcessor:
    """Handle data processing for Excel files with robust error handling and NaN management."""
    
    def __init__(self):
        self.required_daily_columns = [
            'Symbol', 'Name', 'Last Sale', 'Net Change', '% Change', 
            'Market Cap', 'Country', 'IPO Year', 'Volume', 'Sector', 'Industry'
        ]
        
        self.required_historical_columns = [
            'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'
        ]
    
    def process_daily_data(self, file) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Process daily trading data with comprehensive data quality reporting."""
        try:
            # Read file based on extension with enhanced CSV support
            if file.name.endswith('.csv'):
                # Try different CSV parsing options
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file, encoding='latin-1')
                    except:
                        df = pd.read_csv(file, encoding='cp1252')
                except pd.errors.ParserError:
                    # Try with semicolon separator for European CSV format
                    df = pd.read_csv(file, sep=';', encoding='utf-8')
            else:
                df = pd.read_excel(file)
            
            # Initialize quality report
            quality_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_columns': [],
                'data_quality_issues': {},
                'cleaned_rows': 0,
                'nan_handling_summary': {}
            }
            
            # Debug: Show file information
            st.info(f"File loaded: {len(df)} rows, {len(df.columns)} columns")
            st.info(f"Columns detected: {list(df.columns)}")
            
            # Show first few rows for debugging
            if len(df) > 0:
                st.subheader("Raw Data Preview (First 3 rows)")
                st.dataframe(df.head(3), use_container_width=True)
            
            # Smart column matching with flexible alternatives
            column_mapping = {}
            missing_cols = []
            
            # Define alternative column names for better matching
            column_alternatives = {
                'Symbol': ['symbol', 'ticker', 'stock_symbol', 'company_symbol', 'code'],
                'Name': ['name', 'company_name', 'company', 'stock_name', 'security_name'],
                'Last Sale': ['last_sale', 'price', 'last_price', 'close_price', 'current_price', 'last', 'close'],
                'Net Change': ['net_change', 'change', 'price_change', 'net_chg', 'chg'],
                '% Change': ['change', 'percent_change', 'pct_change', '% change', 'change_percent', 'pct_chg', '%_change'],
                'Market Cap': ['market_cap', 'marketcap', 'market_capitalization', 'cap', 'mkt_cap'],
                'Country': ['country', 'nation', 'location', 'domicile'],
                'IPO Year': ['ipo_year', 'ipo', 'year', 'listing_year', 'ipo_date'],
                'Volume': ['volume', 'trade_volume', 'trading_volume', 'vol'],
                'Sector': ['sector', 'industry_sector', 'business_sector', 'gics_sector'],
                'Industry': ['industry', 'sub_industry', 'business_industry', 'gics_industry']
            }
            
            for req_col in self.required_daily_columns:
                found = False
                # Try exact match first (case insensitive)
                for df_col in df.columns:
                    if req_col.lower() == df_col.lower().strip():
                        column_mapping[req_col] = df_col
                        found = True
                        break
                
                # Try alternative names if exact match not found
                if not found and req_col in column_alternatives:
                    for alt_name in column_alternatives[req_col]:
                        for df_col in df.columns:
                            if alt_name.lower() == df_col.lower().strip():
                                column_mapping[req_col] = df_col
                                found = True
                                break
                        if found:
                            break
                
                # Try partial match for common patterns
                if not found:
                    for df_col in df.columns:
                        df_col_clean = df_col.lower().strip()
                        req_col_clean = req_col.lower()
                        if (req_col_clean in df_col_clean or df_col_clean in req_col_clean) and len(df_col_clean) > 2:
                            column_mapping[req_col] = df_col
                            found = True
                            break
                
                if not found:
                    missing_cols.append(req_col)
            
            quality_report['missing_columns'] = missing_cols
            quality_report['column_mapping'] = column_mapping
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Available columns: {list(df.columns)}")
                
                # Try to suggest alternative column mappings
                st.subheader("Suggested Column Mappings")
                for missing_col in missing_cols:
                    similar_cols = [col for col in df.columns if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
                    if similar_cols:
                        st.info(f"For '{missing_col}', consider: {similar_cols}")
                
                # If most columns are missing, try to work with available data
                if len(missing_cols) >= len(self.required_daily_columns) - 2:
                    st.warning("Most required columns are missing. Attempting to work with available data...")
                    # Use only available columns that match
                    available_required_cols = [col for col in self.required_daily_columns if col not in missing_cols]
                    if available_required_cols:
                        df = df[available_required_cols].copy()
                        st.info(f"Working with available columns: {available_required_cols}")
                    else:
                        return None, quality_report
                else:
                    return None, quality_report
            
            # Rename columns to standard format
            df = df.rename(columns={v: k for k, v in column_mapping.items()})
            
            # Select only required columns
            df = df[self.required_daily_columns].copy()
            
            # Handle data type conversions and NaN values
            df = self._clean_daily_data(df, quality_report)
            
            quality_report['cleaned_rows'] = len(df)
            
            return df, quality_report
            
        except Exception as e:
            st.error(f"Error processing daily data: {str(e)}")
            return None, {'error': str(e)}
    
    def process_historical_data(self, file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Process historical price data with date handling."""
        try:
            # Read file based on extension with enhanced CSV support
            if file.name.endswith('.csv'):
                # Try different CSV parsing options
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file, encoding='latin-1')
                    except:
                        df = pd.read_csv(file, encoding='cp1252')
                except pd.errors.ParserError:
                    # Try with semicolon separator for European CSV format
                    df = pd.read_csv(file, sep=';', encoding='utf-8')
            else:
                df = pd.read_excel(file)
            
            # Check for required columns
            missing_cols = [col for col in self.required_historical_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in historical data: {missing_cols}")
                return None, None
            
            # Select only required columns
            df = df[self.required_historical_columns].copy()
            
            # Handle datetime column
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df = df.set_index('Datetime')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date')
            elif df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                except:
                    pass
            
            # Clean numerical data
            numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            for col in numerical_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with too many NaN values
            df = df.dropna(subset=['Close', 'High', 'Low', 'Open'], how='any')
            
            # Fill remaining NaN values
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(df['Volume'].median())
            if 'Dividends' in df.columns:
                df['Dividends'] = df['Dividends'].fillna(0)
            if 'Stock Splits' in df.columns:
                df['Stock Splits'] = df['Stock Splits'].fillna(0)
            
            # Sort by date if index is datetime
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
            
            # Try to extract symbol from filename or sheet name
            symbol = self._extract_symbol_from_file(file)
            
            return df, symbol
            
        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")
            return None, None
    
    def _clean_daily_data(self, df: pd.DataFrame, quality_report: Dict) -> pd.DataFrame:
        """Clean daily trading data with comprehensive NaN handling."""
        
        # Handle numerical columns
        numerical_columns = ['Last Sale', 'Net Change', '% Change', 'Market Cap', 'Volume']
        
        for col in numerical_columns:
            original_nan_count = df[col].isna().sum()
            
            # Convert to numeric, handling various formats
            if col == 'Market Cap':
                df[col] = self._convert_market_cap(df[col])
            elif col == '% Change':
                df[col] = self._convert_percentage(df[col])
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values based on column type
            if col == 'Last Sale':
                df[col] = df[col].fillna(df[col].median())
            elif col == 'Net Change':
                df[col] = df[col].fillna(0)
            elif col == '% Change':
                df[col] = df[col].fillna(0)
            elif col == 'Market Cap':
                df[col] = df[col].fillna(df[col].median())
            elif col == 'Volume':
                df[col] = df[col].fillna(df[col].median())
            
            final_nan_count = df[col].isna().sum()
            
            quality_report['nan_handling_summary'][col] = {
                'original_nan_count': int(original_nan_count),
                'final_nan_count': int(final_nan_count),
                'fill_method': 'median' if col in ['Last Sale', 'Market Cap', 'Volume'] else 'zero'
            }
        
        # Handle categorical columns
        categorical_columns = ['Symbol', 'Name', 'Country', 'Sector', 'Industry']
        
        for col in categorical_columns:
            original_nan_count = df[col].isna().sum()
            df[col] = df[col].fillna('Unknown')
            final_nan_count = df[col].isna().sum()
            
            quality_report['nan_handling_summary'][col] = {
                'original_nan_count': int(original_nan_count),
                'final_nan_count': int(final_nan_count),
                'fill_method': 'Unknown'
            }
        
        # Handle IPO Year
        df['IPO Year'] = pd.to_numeric(df['IPO Year'], errors='coerce')
        df['IPO Year'] = df['IPO Year'].fillna(df['IPO Year'].median())
        
        # Check if critical columns exist before dropping rows
        available_critical_columns = [col for col in ['Symbol', 'Last Sale'] if col in df.columns]
        
        if available_critical_columns:
            # Only drop rows where ALL available critical columns are missing
            rows_before_drop = len(df)
            df = df.dropna(subset=available_critical_columns, how='all')
            rows_after_drop = len(df)
            
            if rows_before_drop > 0 and rows_after_drop == 0:
                st.warning(f"All {rows_before_drop} rows were removed due to missing critical data in columns: {available_critical_columns}")
                st.info("This might be due to column name mismatches or data format issues.")
            elif rows_before_drop != rows_after_drop:
                st.info(f"Removed {rows_before_drop - rows_after_drop} rows with missing critical data.")
        else:
            st.warning("No critical columns (Symbol, Last Sale) found in the data. Proceeding without dropping rows.")
        
        # Data validation
        quality_report['data_quality_issues'] = self._validate_data_quality(df)
        
        return df
    
    def _convert_market_cap(self, series: pd.Series) -> pd.Series:
        """Convert market cap strings to numerical values."""
        def convert_value(val):
            if pd.isna(val):
                return np.nan
            
            if isinstance(val, (int, float)):
                return float(val)
            
            val = str(val).upper().replace('$', '').replace(',', '').strip()
            
            if val == '' or val == 'N/A':
                return np.nan
            
            try:
                if 'B' in val:
                    return float(val.replace('B', '')) * 1e9
                elif 'M' in val:
                    return float(val.replace('M', '')) * 1e6
                elif 'K' in val:
                    return float(val.replace('K', '')) * 1e3
                else:
                    return float(val)
            except:
                return np.nan
        
        return series.apply(convert_value)
    
    def _convert_percentage(self, series: pd.Series) -> pd.Series:
        """Convert percentage strings to numerical values."""
        def convert_value(val):
            if pd.isna(val):
                return np.nan
            
            if isinstance(val, (int, float)):
                return float(val)
            
            val = str(val).replace('%', '').strip()
            
            try:
                return float(val)
            except:
                return np.nan
        
        return series.apply(convert_value)
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and identify potential issues."""
        issues = {}
        
        # Calculate valid symbols (non-empty, non-null Symbol entries)
        if 'Symbol' in df.columns:
            valid_symbols = df['Symbol'].notna() & (df['Symbol'] != '') & (df['Symbol'] != 'nan')
            issues['valid_symbols'] = int(valid_symbols.sum())
        else:
            issues['valid_symbols'] = 0
        
        # Check for negative volumes
        negative_volumes = (df['Volume'] < 0).sum()
        if negative_volumes > 0:
            issues['negative_volumes'] = int(negative_volumes)
        
        # Check for zero market caps
        zero_market_caps = (df['Market Cap'] <= 0).sum()
        if zero_market_caps > 0:
            issues['zero_market_caps'] = int(zero_market_caps)
        
        # Check for extreme percentage changes
        extreme_changes = (abs(df['% Change']) > 100).sum()
        if extreme_changes > 0:
            issues['extreme_percentage_changes'] = int(extreme_changes)
        
        # Check for future IPO years
        current_year = pd.Timestamp.now().year
        future_ipos = (df['IPO Year'] > current_year).sum()
        if future_ipos > 0:
            issues['future_ipo_years'] = int(future_ipos)
        
        return issues
    
    def _extract_symbol_from_file(self, file) -> Optional[str]:
        """Try to extract stock symbol from filename."""
        try:
            filename = file.name
            # Common patterns for stock symbols in filenames
            import re
            
            # Look for patterns like AAPL, MSFT, etc.
            pattern = r'[A-Z]{1,5}'
            matches = re.findall(pattern, filename.upper())
            
            if matches:
                # Return the first match that looks like a stock symbol
                for match in matches:
                    if 2 <= len(match) <= 5:
                        return match
            
            return None
        except:
            return None
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive data summary."""
        summary = {
            'total_records': len(df),
            'date_range': {},
            'numerical_summary': {},
            'categorical_summary': {}
        }
        
        # Date range (if datetime index)
        if pd.api.types.is_datetime64_any_dtype(df.index):
            summary['date_range'] = {
                'start_date': str(df.index.min().date()),
                'end_date': str(df.index.max().date()),
                'total_days': (df.index.max() - df.index.min()).days
            }
        
        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            summary['numerical_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isna().sum())
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': int(df[col].nunique()),
                'most_common': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A',
                'null_count': int(df[col].isna().sum())
            }
        
        return summary
