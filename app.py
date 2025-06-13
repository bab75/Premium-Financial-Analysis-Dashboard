import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import re
import traceback

# Import utility modules
from utils.data_processor import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.visualizations import Visualizations
from utils.predictions import PricePredictions
from utils.analytics import Analytics
from utils.comparative_analysis import ComparativeAnalysis
from utils.enhanced_processor import EnhancedDataProcessor
from utils.fixed_processor import FixedDataProcessor
from utils.comprehensive_fix import ComprehensiveFix
from utils.risk_gauge import RiskGauge
from utils.chart_formatter import ChartFormatter

warnings.filterwarnings('ignore')

# Page configuration - MUST be first
st.set_page_config(
    page_title="Premium Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize or reset session state variables."""
    session_keys = [
        'current_data', 'previous_data', 'historical_data', 'selected_symbol',
        'data_quality_report', 'comparative_analysis', 'yfinance_data', 'clear_trigger'
    ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

initialize_session_state()

def clean_numeric_column(series):
    """Clean a column to ensure numeric values, removing common non-numeric characters."""
    original_series = series.copy()
    if series.dtype == 'object':
        # Replace common invalid values with NaN
        invalid_values = ['N/A', 'NA', '--', '', 'null', 'None']
        series = series.replace(invalid_values, np.nan)
        # Remove $, commas, %, and other non-numeric characters
        series = series.str.replace(r'[\$,%]', '', regex=True).str.strip()
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
    return series, original_series

def clear_session_state():
    """Clear session state variables except navigation."""
    keys_to_clear = [key for key in st.session_state.keys() if key not in ['navigation']]
    for key in keys_to_clear:
        st.session_state[key] = None
    st.session_state['clear_message'] = "✅ All data cleared! You can now upload new files."
    st.rerun()

def validate_columns(df, required_cols):
    """Validate if required columns exist and contain valid data."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}. Found: {', '.join(df.columns)}"
    
    # Check if Last Sale has any potentially numeric values
    if 'Last Sale' in df.columns:
        sample = df['Last Sale'].head().astype(str).tolist()
        has_potential_numbers = any(re.match(r'^-?\d*\.?\d+$', str(x).strip()) for x in sample if pd.notna(x))
        if not has_potential_numbers:
            return False, f"Column 'Last Sale' contains no potentially numeric data. Sample: {sample}"
    return True, ""

def data_upload_section():
    """Enhanced Data Upload & Processing section."""
    st.header("📁 Enhanced Data Upload & Processing")
    st.markdown("Upload your stock data files for comprehensive Phase 1 & Phase 2 analysis")
    
    # Display clear message if set
    if 'clear_message' in st.session_state:
        st.success(st.session_state['clear_message'])
        del st.session_state['clear_message']
    
    processor = DataProcessor()
    
    # Current Data Upload
    st.subheader("📊 Current Stock Data")
    current_file = st.file_uploader(
        "Upload Current Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="current_data_file",
        help="Upload your current stock trading data with columns like Symbol, Last Sale, Net Change, % Change, Sector, Industry"
    )
    
    if current_file is not None:
        try:
            with st.spinner("Processing current stock data..."):
                current_data, quality_report = processor.process_daily_data(current_file)
                
                if current_data is not None:
                    # Validate required columns
                    required_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', 'Industry']
                    is_valid, error_msg = validate_columns(current_data, required_columns)
                    if not is_valid:
                        st.error(error_msg)
                        st.info("Please ensure the file has the correct columns and 'Last Sale' contains numeric values (e.g., 100.50).")
                        return
                    
                    # Clean and validate numeric columns
                    numeric_cols = ['Last Sale', 'Net Change', '% Change']
                    for col in numeric_cols:
                        try:
                            current_data[col], original_col = clean_numeric_column(current_data[col])
                            if current_data[col].isna().all():
                                st.error(f"Column '{col}' contains no valid numeric data after cleaning.")
                                st.write(f"Sample raw values in '{col}' (first 5 rows): {original_col.head().tolist()}")
                                st.info("Please ensure the column contains numeric values (e.g., 100.50) without symbols like $, text like 'N/A', or empty cells. Check file encoding or column names.")
                                st.session_state.current_data = current_data  # Store partial data
                                return
                            if current_data[col].isna().sum() > 0:
                                st.warning(f"Column '{col}' has {current_data[col].isna().sum()} missing or invalid values, which may affect analysis.")
                        except Exception as e:
                            st.error(f"Error processing column '{col}': {str(e)}")
                            return
                    
                    st.session_state.current_data = current_data
                    st.session_state.data_quality_report = quality_report
                    
                    st.success(f"✅ Current data loaded successfully! ({len(current_data)} stocks)")
                    
                    # Data quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Stocks", len(current_data))
                    with col2:
                        valid_count = len(current_data[current_data['Symbol'].notna() & 
                                                      (current_data['Symbol'].astype(str).str.strip() != '')])
                        st.metric("Valid Symbols", valid_count)
                    with col3:
                        st.metric("Data Completeness", f"{quality_report.get('completeness_score', 0):.1f}%")
                    with col4:
                        st.metric("Quality Score", f"{quality_report.get('overall_quality', 0):.1f}/10")
                    
                    with st.expander("📋 Sample Data Preview"):
                        st.dataframe(current_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process current data file. Please check the format (CSV/Excel) and column names.")
                    
        except Exception as e:
            st.error(f"Error processing current data: {str(e)}")
            st.write(traceback.format_exc())
    
    # Previous Data Upload
    st.subheader("📈 Previous Stock Data")
    previous_file = st.file_uploader(
        "Upload Previous Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="previous_data_file",
        help="Upload previous period stock data for Phase 1 comparative analysis"
    )
    
    if previous_file is not None:
        try:
            with st.spinner("Processing previous stock data..."):
                previous_data, prev_quality_report = processor.process_daily_data(previous_file)
                
                if previous_data is not None:
                    required_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', 'Industry']
                    is_valid, error_msg = validate_columns(previous_data, required_columns)
                    if not is_valid:
                        st.error(error_msg)
                        st.info("Please ensure the file has the correct columns and 'Last Sale' contains numeric values (e.g., 100.50).")
                        return
                    
                    numeric_cols = ['Last Sale', 'Net Change', '% Change']
                    for col in numeric_cols:
                        try:
                            previous_data[col], original_col = clean_numeric_column(previous_data[col])
                            if previous_data[col].isna().all():
                                st.error(f"Column '{col}' contains no valid numeric data after cleaning in previous data.")
                                st.write(f"Sample raw values in '{col}' (first 5 rows): {original_col.head().tolist()}")
                                st.info("Please ensure the column contains numeric values (e.g., 100.50) without symbols like $, text like 'N/A', or empty cells.")
                                st.session_state.previous_data = previous_data
                                return
                            if previous_data[col].isna().sum() > 0:
                                st.warning(f"Column '{col}' has {previous_data[col].isna().sum()} missing or invalid values.")
                        except Exception as e:
                            st.error(f"Error processing column '{col}' in previous data: {str(e)}")
                            return
                    
                    st.session_state.previous_data = previous_data
                    st.success(f"✅ Previous data loaded successfully! ({len(previous_data)} stocks)")
                    
                    with st.expander("📋 Previous Data Preview"):
                        st.dataframe(previous_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process previous data file. Check format and column names.")
                    
        except Exception as e:
            st.error(f"Error processing previous data: {str(e)}")
    
    # Process Button
    if st.session_state.current_data is not None and st.session_state.previous_data is not None:
        st.success("🎉 Both datasets are ready for analysis!")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.success("✅ Analysis ready! Navigate to Phase 1 to view results.")
                st.balloons()
    
    # Historical Data Upload
    st.subheader("📉 Historical Price Data (Optional)")
    historical_file = st.file_uploader(
        "Upload Historical Price Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="historical_data_file",
        help="Upload historical price data with Date, Open, High, Low, Close, Volume columns"
    )
    
    if historical_file is not None:
        try:
            with st.spinner("Processing historical data..."):
                historical_data, extracted_symbol = processor.process_historical_data(historical_file)
                if historical_data is not None:
                    st.session_state.historical_data = historical_data
                    if extracted_symbol:
                        st.session_state.selected_symbol = extracted_symbol
                    st.success(f"✅ Historical data loaded successfully! ({len(historical_data)} data points)")
                    with st.expander("📋 Historical Data Preview"):
                        st.dataframe(historical_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process historical data file. Check format.")
        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")
    
    # Data Status Dashboard
    if st.session_state.current_data is not None or st.session_state.previous_data is not None:
        st.markdown("---")
        st.subheader("📊 Data Status Dashboard")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown(
                f"<div class='{'success' if st.session_state.current_data is not None else 'warning'}-card'>"
                f"<h4>{'✅' if st.session_state.current_data is not None else '⏳'} Current Data</h4>"
                f"<p>{'Ready for analysis' if st.session_state.current_data is not None else 'Not uploaded'}</p></div>",
                unsafe_allow_html=True
            )
        with status_col2:
            st.markdown(
                f"<div class='{'success' if st.session_state.previous_data is not None else 'warning'}-card'>"
                f"<h4>{'✅' if st.session_state.previous_data is not None else '⏳'} Previous Data</h4>"
                f"<p>{'Ready for Phase 1' if st.session_state.previous_data is not None else 'Upload for Phase 1'}</p></div>",
                unsafe_allow_html=True
            )
        with status_col3:
            st.markdown(
                f"<div class='{'success' if st.session_state.historical_data is not None else 'metric'}-card'>"
                f"<h4>{'✅' if st.session_state.historical_data is not None else '📈'} Historical Data</h4>"
                f"<p>{'Ready for analysis' if st.session_state.historical_data is not None else 'Optional'}</p></div>",
                unsafe_allow_html=True
            )

def phase1_comparative_analysis_section():
    """Phase 1: Comparative analysis between current and previous stock data."""
    st.header("📊 Phase 1: Comparative Analysis")
    st.markdown("Compare current vs previous stock data")
    
    if st.session_state.current_data is None or st.session_state.previous_data is None:
        st.warning("⚠️ Upload both current and previous stock data files in the Data Upload tab.")
        return
    
    try:
        with st.spinner("Performing comparative analysis..."):
            comp_analysis = ComparativeAnalysis(st.session_state.current_data, st.session_state.previous_data)
            st.session_state.comparative_analysis = comp_analysis
            merged_data = comp_analysis.merged_data
            
            if merged_data.empty:
                st.error("No matching stocks found. Check Symbol columns for consistency.")
                return
            
            required_cols = ['Last Sale_curr', 'Last Sale_prev', '% Change_calc', 'Profit/Loss']
            missing_cols = [col for col in required_cols if col not in merged_data.columns]
            if missing_cols:
                st.error(f"Missing columns in merged data: {', '.join(missing_cols)}")
                return
            
            if merged_data['% Change_calc'].isna().all():
                st.error("No valid price change data. Check 'Last Sale' values in input files.")
                st.write("Sample merged data for debugging:")
                st.dataframe(merged_data[['Symbol', 'Last Sale_curr', 'Last Sale_prev', '% Change_calc']].head())
                return
        
        st.subheader("📈 Performance Summary")
        summary = comp_analysis.get_performance_summary()
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks Analyzed", summary.get('total_stocks', 0))
            with col2:
                st.metric("Average Price Change", f"{summary.get('avg_change', 0):.2f}%")
            with col3:
                st.metric("Gainers", summary.get('gainers', 0))
            with col4:
                st.metric("Losers", summary.get('losers', 0))
        
        performers_col1, performers_col2 = st.columns(2)
        with performers_col1:
            st.subheader("🏆 Top 5 Performers")
            valid_data = merged_data.dropna(subset=['% Change_calc'])
            if not valid_data.empty:
                top_performers = valid_data.nlargest(5, '% Change_calc')
                for _, row in top_performers.iterrows():
                    st.success(f"🟢 **{row['Symbol']}**: {row['% Change_calc']:.2f}%")
            else:
                st.info("No top performers data available")
        
        with performers_col2:
            st.subheader("📉 Bottom 5 Performers")
            if not valid_data.empty:
                bottom_performers = valid_data.nsmallest(5, '% Change_calc')
                for _, row in bottom_performers.iterrows():
                    st.error(f"🔴 **{row['Symbol']}**: {row['% Change_calc']:.2f}%")
            else:
                st.info("No bottom performers data available")
        
        st.subheader("📊 Performance Dashboard")
        try:
            dashboard_fig = comp_analysis.create_performance_dashboard()
            if dashboard_fig and dashboard_fig.data:
                st.plotly_chart(dashboard_fig, use_container_width=True)
            else:
                st.info("Performance dashboard unavailable due to insufficient data.")
        except Exception as e:
            st.info(f"Dashboard unavailable: {str(e)}")
        
        st.subheader("🎯 Risk Assessment")
        risk_gauge = RiskGauge()
        try:
            if len(merged_data) > 1 and '% Change_calc' in merged_data.columns:
                price_volatility = float(merged_data['% Change_calc'].std())
                avg_performance = float(merged_data['% Change_calc'].mean())
                risk_data = {
                    'risk_score': min(100, max(0, price_volatility * 3)),
                    'volatility': min(100, max(0, price_volatility * 2)),
                    'performance': min(100, max(0, avg_performance + 50))
                }
                gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                with gauge_col1:
                    st.plotly_chart(risk_gauge.create_risk_gauge(risk_data['risk_score'], "Market Risk"), use_container_width=True)
                with gauge_col2:
                    st.plotly_chart(risk_gauge.create_volatility_gauge(risk_data['volatility']), use_container_width=True)
                with gauge_col3:
                    st.plotly_chart(risk_gauge.create_performance_gauge(avg_performance), use_container_width=True)
            else:
                st.info("Risk assessment unavailable.")
        except Exception as e:
            st.info(f"Risk assessment unavailable: {str(e)}")
        
        st.subheader("📊 Detailed Stock Analysis")
        if not valid_data.empty:
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                performance_filter = st.selectbox(
                    "Filter by Performance",
                    ["All Stocks", "Gainers Only", "Losers Only", "Top 10", "Bottom 10"]
                )
            with filter_col2:
                min_change = st.number_input("Min Change %", value=float(valid_data['% Change_calc'].min()))
            with filter_col3:
                max_change = st.number_input("Max Change %", value=float(valid_data['% Change_calc'].max()))
            
            filtered_df = valid_data[
                (valid_data['% Change_calc'] >= min_change) & 
                (valid_data['% Change_calc'] <= max_change)
            ].copy()
            
            if performance_filter == "Gainers Only":
                filtered_df = filtered_df[filtered_df['% Change_calc'] > 0]
            elif performance_filter == "Losers Only":
                filtered_df = filtered_df[filtered_df['% Change_calc'] < 0]
            elif performance_filter == "Top 10":
                filtered_df = filtered_df.nlargest(10, '% Change_calc')
            elif performance_filter == "Bottom 10":
                filtered_df = filtered_df.nsmallest(10, '% Change_calc')
            
            if not filtered_df.empty:
                filtered_df = filtered_df.sort_values('% Change_calc', ascending=False)
                display_columns = ['Symbol', '% Change_calc', 'Profit/Loss', 'Last Sale_curr', 'Last Sale_prev']
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                st.dataframe(filtered_df[available_columns], use_container_width=True, height=400)
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name="filtered_stocks.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No stocks match the filter criteria")
        else:
            st.warning("No valid price change data available")
        
        st.subheader("🏭 Sector Analysis")
        sector_analysis = comp_analysis.get_sector_analysis()
        if not sector_analysis.empty:
            st.dataframe(sector_analysis, use_container_width=True)
        else:
            st.info("Sector analysis unavailable")
            
    except Exception as e:
        st.error(f"Error in Phase 1 analysis: {str(e)}")

def phase2_deep_analysis_section():
    """Phase 2: Deep analysis with yfinance integration."""
    st.header("📈 Phase 2: Deep Stock Analysis")
    st.markdown("Technical analysis with yfinance data")
    
    available_stocks = []
    if st.session_state.current_data is not None and 'Symbol' in st.session_state.current_data.columns:
        available_stocks.extend(st.session_state.current_data['Symbol'].dropna().unique().tolist())
    if st.session_state.previous_data is not None and 'Symbol' in st.session_state.previous_data.columns:
        available_stocks.extend([s for s in st.session_state.previous_data['Symbol'].dropna().unique().tolist() if s not in available_stocks])
    
    if not available_stocks:
        st.warning("⚠️ Upload stock data in the Data Upload tab.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_stock = st.selectbox("Select Stock", options=available_stocks)
    with col2:
        analysis_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if selected_stock and st.button("🔄 Fetch Data", type="primary"):
        try:
            with st.spinner(f"Fetching data for {selected_stock}..."):
                ticker = yf.Ticker(selected_stock)
                hist_data = ticker.history(period=analysis_period)
                if hist_data.empty:
                    st.error(f"No data available for {selected_stock}")
                    return
                
                st.session_state.yfinance_data = hist_data
                st.session_state.selected_symbol = selected_stock
                
                info = ticker.info
                company_name = info.get('longName', selected_stock)
                st.markdown(f"<h3>📋 {company_name} ({selected_stock})</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${hist_data['Close'].iloc[-1]:.2f}")
                with col2:
                    st.metric("Sector", info.get('sector', 'Unknown'))
                with col3:
                    st.metric("Industry", info.get('industry', 'Unknown'))
                
                hist_data_clean = hist_data.copy().reset_index()
                if 'Adj Close' not in hist_data_clean.columns:
                    hist_data_clean['Adj Close'] = hist_data_clean['Close']
                
                st.subheader("📊 Price Visualizations")
                viz = Visualizations(historical_data=hist_data_clean)
                st.plotly_chart(viz.create_candlestick_chart(), use_container_width=True)
                
                if len(hist_data_clean) > 50:
                    st.subheader("⚙️ Technical Indicators")
                    tech_indicators = TechnicalIndicators(hist_data_clean)
                    st.plotly_chart(tech_indicators.create_moving_averages_chart(), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

def advanced_analytics_section():
    """Advanced Analytics: Predictions and Insights."""
    st.header("🔮 Advanced Analytics & Predictions")
    
    data_source = st.session_state.yfinance_data if st.session_state.yfinance_data is not None else st.session_state.historical_data
    if data_source is None or data_source.empty:
        st.warning("⚠️ Upload historical data or fetch yfinance data in Phase 2.")
        return
    
    data_clean = data_source.copy().reset_index()
    if 'Adj Close' not in data_clean.columns:
        data_clean['Adj Close'] = data_clean['Close']
    for col in ['Dividends', 'Stock Splits']:
        if col not in data_clean.columns:
            data_clean[col] = 0
    if 'Date' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Date': 'Datetime'})
    
    pred_tab, viz_tab = st.tabs(["🔮 Price Predictions", "📊 Visualizations"])
    
    with pred_tab:
        st.subheader("🔮 Price Predictions")
        if len(data_clean) > 50:
            predictions = PricePredictions(data_clean)
            pred_days = st.slider("Prediction Days", 1, 30, 7)
            pred_method = st.selectbox("Prediction Method", ["technical_analysis", "linear_trend"])
            if st.button("Generate Predictions"):
                pred_prices = predictions.predict_prices(pred_days, pred_method)
                if pred_prices:
                    pred_chart = predictions.create_prediction_chart(pred_prices, pred_days)
                    st.plotly_chart(pred_chart, use_container_width=True)
                    pred_df = pd.DataFrame({
                        'Date': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)],
                        'Predicted Price': [f"${p:.2f}" for p in pred_prices]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                else:
                    st.error("Unable to generate predictions.")
        else:
            st.warning("Need >50 data points for predictions.")
    
    with viz_tab:
        st.subheader("📊 Advanced Visualizations")
        viz = Visualizations(historical_data=data_clean)
        st.plotly_chart(viz.create_price_trends_chart(), use_container_width=True)

def main():
    """Main application layout."""
    st.sidebar.title("Premium Financial Analysis Dashboard")
    page = st.sidebar.radio(
        "Select Phase",
        ["Data Upload", "Phase 1: Comparative Analysis", "Phase 2: Deep Analysis", "Advanced Analytics"],
        key="navigation"
    )
    
    st.sidebar.markdown("---")
    with st.sidebar.form(key="clear_form"):
        clear_submitted = st.form_submit_button("🗑️ Clear Analysis")
        if clear_submitted:
            clear_session_state()
    
    if page == "Data Upload":
        data_upload_section()
    elif page == "Phase 1: Comparative Analysis":
        phase1_comparative_analysis_section()
    elif page == "Phase 2: Deep Analysis":
        phase2_deep_analysis_section()
    elif page == "Advanced Analytics":
        advanced_analytics_section()

if __name__ == "__main__":
    main()
