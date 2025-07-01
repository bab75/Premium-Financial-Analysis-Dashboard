import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime as dt
import warnings

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
from utils.robust_processor import RobustProcessor
from utils.enhanced_comparative_analysis import EnhancedComparativeAnalysis
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Premium Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar clear functionality
with st.sidebar:
    st.header("🔧 Controls")
    if st.button("🗑️ Clear All Data", type="secondary", help="Refresh page to clear all data and start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
       
# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'selected_stock_symbol' not in st.session_state:
    st.session_state.selected_stock_symbol = None
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = None
if 'comparative_analysis' not in st.session_state:
    st.session_state.comparative_analysis = None
if 'yfinance_data' not in st.session_state:
    st.session_state.yfinance_data = None

def data_upload_section():
    st.header("📁 Enhanced Data Upload & Processing")
    st.markdown("Upload your stock data files for comprehensive Phase 1 & Phase 2 analysis")

    if st.button("🗑️ Clear All Uploaded Files", type="secondary"):
        keys_to_clear = [
            'current_data', 'previous_data', 'daily_data', 'historical_data',
            'comparative_analysis', 'yfinance_data', 'data_quality_report',
            'selected_symbol', 'current_data_file', 'previous_data_file', 
            'historical_data_file'
        ]
        cleared_count = 0
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                cleared_count += 1
        # Clear file uploader states
        for key in list(st.session_state.keys()):
            key_str = str(key).lower()
            if 'file' in key_str or 'data' in key_str:
                if key not in keys_to_clear:
                    del st.session_state[key]
                    cleared_count += 1
        st.success(f"Successfully cleared {cleared_count} data items and uploaded files!")
        st.rerun()

    robust_processor = RobustProcessor()

    st.subheader("📊 Current Stock Data")
    current_file = st.file_uploader(
        "Upload Current Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="current_data_file",
        help="Upload your current stock trading data with columns like Symbol, Name, Last Sale, % Change, etc."
    )

    if current_file is not None:
        try:
            with st.spinner("Processing current stock data..."):
                current_data, quality_report = robust_processor.process_uploaded_data(current_file)
                if current_data is not None:
                    st.session_state.current_data = current_data
                    st.session_state.data_quality_report = quality_report
                    st.success(f"✅ Current data loaded successfully! ({len(current_data)} stocks)")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Stocks", len(current_data))
                    with col2:
                        valid_count = len(current_data[current_data['Symbol'].notna() & 
                                                     (current_data['Symbol'].astype(str).str.strip() != '') & 
                                                     (current_data['Symbol'].astype(str).str.strip() != 'nan')])
                        st.metric("Valid Symbols", valid_count)
                    with col3:
                        st.metric("Data Completeness", f"{quality_report.get('completeness_score', 0):.1f}%")
                    with col4:
                        st.metric("Quality Score", f"{quality_report.get('overall_quality', 0):.1f}/10")
                    with st.expander("📋 Sample Data Preview"):
                        st.dataframe(current_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process current data file. Please check the format.")
        except Exception as e:
            st.error(f"Error processing current data: {str(e)}")

    st.subheader("📈 Previous Stock Data (For Comparative Analysis)")
    previous_file = st.file_uploader(
        "Upload Previous Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="previous_data_file",
        help="Upload previous period stock data for Phase 1 comparative analysis"
    )

    if previous_file is not None:
        try:
            with st.spinner("Processing previous stock data..."):
                previous_data, prev_quality_report = robust_processor.process_uploaded_data(previous_file)
                if previous_data is not None:
                    st.session_state.previous_data = previous_data
                    st.success(f"✅ Previous data loaded successfully! ({len(previous_data)} stocks)")
                    with st.expander("📋 Previous Data Preview"):
                        st.dataframe(previous_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process previous data file. Please check the format.")
        except Exception as e:
            st.error(f"Error processing previous data: {str(e)}")

    if ('current_data' in st.session_state and 'previous_data' in st.session_state):
        st.success("🎉 Both datasets are ready for analysis!")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.success("✅ Analysis ready! Navigate to Phase 1: Comparative Analysis to see results.")
                st.balloons()

    st.subheader("📉 Historical Price Data (Optional)")
    st.info("Upload historical price data to enable technical analysis and visualizations without yfinance.")
    historical_file = st.file_uploader(
        "Upload Historical Price Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="historical_data_file",
        help="Upload historical price data with Date, Open, High, Low, Close, Volume columns"
    )
    
    if historical_file is not None:
        try:
            with st.spinner("Processing historical data..."):
                # Temporary: Load file directly to debug
                if historical_file.name.endswith('.csv'):
                    historical_data = pd.read_csv(historical_file, encoding='utf-8', sep=',', engine='python')
                else:
                    historical_data = pd.read_excel(historical_file)
                st.write(f"Directly loaded columns: {list(historical_data.columns)}")
                extracted_symbol = None  # Placeholder for symbol extraction
                
                if historical_data is not None:
                    # Clean column names (remove spaces, normalize case)
                    historical_data.columns = [col.strip() for col in historical_data.columns]
                    st.write(f"Cleaned columns: {list(historical_data.columns)}")
                    
                    # Rename 'Date' or variations to 'Datetime'
                    column_mapping = {col: col for col in historical_data.columns}
                    for col in historical_data.columns:
                        if col.lower() in ['date', 'datetime', 'time']:
                            column_mapping[col] = 'Datetime'
                    historical_data = historical_data.rename(columns=column_mapping)
                    
                    # Validate required columns
                    required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_columns if col not in historical_data.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        historical_data['Datetime'] = pd.to_datetime(historical_data['Datetime'], errors='coerce')
                        if historical_data['Datetime'].isna().any():
                            st.warning("Some datetime values could not be parsed and were set to NaN.")
                        historical_data = historical_data.set_index('Datetime')
                        if 'Adj Close' not in historical_data.columns:
                            historical_data['Adj Close'] = historical_data['Close']
                        st.session_state.historical_data = historical_data
                        if extracted_symbol:
                            st.session_state.selected_symbol = extracted_symbol
                        st.success(f"✅ Historical data loaded successfully! ({len(historical_data)} data points)")
                        if extracted_symbol:
                            st.info(f"📊 Detected symbol: {extracted_symbol}")
                        with st.expander("📋 Historical Data Preview"):
                            st.dataframe(historical_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process historical data file. Please check the format.")
        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")

    if st.session_state.current_data is not None or st.session_state.previous_data is not None:
        st.markdown("---")
        st.subheader("📊 Data Status Dashboard")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            if st.session_state.current_data is not None:
                st.markdown("<div class='success-card'><h4>✅ Current Data</h4><p>Ready for analysis</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-card'><h4>⏳ Current Data</h4><p>Not uploaded</p></div>", unsafe_allow_html=True)
        with status_col2:
            if st.session_state.previous_data is not None:
                st.markdown("<div class='success-card'><h4>✅ Previous Data</h4><p>Ready for Phase 1</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-card'><h4>⏳ Previous Data</h4><p>Upload for Phase 1</p></div>", unsafe_allow_html=True)
        with status_col3:
            if st.session_state.historical_data is not None:
                st.markdown("<div class='success-card'><h4>✅ Historical Data</h4><p>Ready for analysis</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-card'><h4>📈 Historical Data</h4><p>Optional for advanced analysis</p></div>", unsafe_allow_html=True)

def phase1_comparative_analysis_section():
    st.header("📊 Phase 1: Comparative Analysis")
    st.markdown("Compare current vs previous stock data for comprehensive market analysis")
    st.info("💡 To clear data or reset the analysis, please refresh the browser page.")
    if st.session_state.current_data is None or st.session_state.previous_data is None:
        st.warning("⚠️ Please upload both current and previous stock data files in the Data Upload tab first.")
        return
    try:
        with st.spinner("Performing enhanced comparative analysis..."):
            comp_analysis = EnhancedComparativeAnalysis(st.session_state.current_data, st.session_state.previous_data)
            if comp_analysis.merged_data is None or comp_analysis.merged_data.empty:
                st.error("No matching stocks found between the two datasets")
                return
            merged_data = comp_analysis.merged_data
        st.subheader("📈 Overall Performance Summary")
        summary = comp_analysis.get_performance_summary()
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks Analyzed", summary.get('total_stocks', 0))
            with col2:
                avg_change = summary.get('avg_change', 0)
                st.metric("Average Price Change", f"{avg_change:.2f}%", delta=f"{avg_change:.2f}%")
            with col3:
                gainers = summary.get('gainers', 0)
                st.metric("Gainers", gainers, delta="positive" if gainers > 0 else None)
            with col4:
                losers = summary.get('losers', 0)
                st.metric("Losers", losers, delta="negative" if losers > 0 else None)
        st.subheader("📈 Interactive Performance Visualization")
        try:
            performance_chart = comp_analysis.create_interactive_performance_chart()
            if performance_chart and hasattr(performance_chart, 'data') and performance_chart.data:
                st.plotly_chart(performance_chart, use_container_width=True)
            else:
                st.info("Performance chart not available")
        except Exception as e:
            st.warning(f"Could not generate performance chart: {str(e)}")
        performers_col1, performers_col2 = st.columns(2)
        with performers_col1:
            st.subheader("🏆 Top 5 Performers")
            if not merged_data.empty and 'Price_Change_Pct' in merged_data.columns:
                try:
                    top_data = merged_data.dropna(subset=['Price_Change_Pct'])
                    if not top_data.empty:
                        top_performers = top_data.nlargest(5, 'Price_Change_Pct')
                        for idx, row in top_performers.iterrows():
                            symbol = row.get('Symbol', 'N/A')
                            change_pct = row.get('Price_Change_Pct', 0)
                            change_amt = row.get('Price_Change', 0)
                            st.success(f"**{symbol}**: {change_pct:.2f}% (${change_amt:.2f})")
                    else:
                        st.info("No top performers data available")
                except Exception as e:
                    st.warning(f"Could not display top performers: {str(e)}")
            else:
                st.info("No top performers data available")
        with performers_col2:
            st.subheader("📉 Bottom 5 Performers")
            if not merged_data.empty and 'Price_Change_Pct' in merged_data.columns:
                try:
                    bottom_data = merged_data.dropna(subset=['Price_Change_Pct'])
                    if not bottom_data.empty:
                        bottom_performers = bottom_data.nsmallest(5, 'Price_Change_Pct')
                        for idx, row in bottom_performers.iterrows():
                            symbol = row.get('Symbol', 'N/A')
                            change_pct = row.get('Price_Change_Pct', 0)
                            change_amt = row.get('Price_Change', 0)
                            st.error(f"**{symbol}**: {change_pct:.2f}% (${change_amt:.2f})")
                    else:
                        st.info("No bottom performers data available")
                except Exception as e:
                    st.warning(f"Could not display bottom performers: {str(e)}")
            else:
                st.info("No bottom performers data available")
        st.subheader("📊 Analysis Summary")
        if not merged_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Dataset Overview")
                st.info(f"Successfully compared {len(merged_data)} stocks")
                if 'Price_Change_Pct' in merged_data.columns:
                    positive_changes = (merged_data['Price_Change_Pct'] > 0).sum()
                    negative_changes = (merged_data['Price_Change_Pct'] < 0).sum()
                    unchanged = (merged_data['Price_Change_Pct'] == 0).sum()
                    st.write(f"- **Gainers:** {positive_changes}")
                    st.write(f"- **Losers:** {negative_changes}")
                    st.write(f"- **Unchanged:** {unchanged}")
            with col2:
                st.subheader("💾 Download Results")
                if st.button("📄 Download Full Analysis (CSV)", type="secondary"):
                    csv_data = merged_data.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv_data,
                        file_name=f"stock_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        st.subheader("🎯 Professional Risk Assessment")
        with st.expander("📚 How to Read Risk Assessment Gauges", expanded=False):
            st.markdown("""
            **Professional Risk Gauges:** These voltage meter-style visualizations provide instant insights into your portfolio's risk profile.
            **Three Key Metrics:**
            - **Market Risk (Left):** Overall risk level based on price volatility patterns
            - **Volatility Index (Center):** Measures price fluctuation intensity over time  
            - **Performance Score (Right):** Composite performance rating across multiple factors
            **Reading the Gauges:**
            - **Green Zone (0-30%):** Conservative, stable investments
            - **Yellow Zone (30-60%):** Moderate risk, balanced approach
            - **Orange Zone (60-80%):** Higher risk, growth-focused
            - **Red Zone (80-100%):** High risk, aggressive investments
            - **Purple Zone (100%):** Extreme risk, highly aggressive investments
            - **Real-time Example:** If Market Risk shows 45%, Volatility shows 35%, and Performance shows 72%, your portfolio has moderate risk with good returns - ideal for balanced investors.
            """)
        risk_gauge = RiskGauge()
        try:
            if merged_data is not None and len(merged_data) > 1 and 'Price_Change_Pct' in merged_data.columns:
                price_volatility = float(merged_data['Price_Change_Pct'].std())
                avg_performance = float(merged_data['Price_Change_Pct'].mean())
                max_gain = float(merged_data['Price_Change_Pct'].max())
                max_loss = float(merged_data['Price_Change_Pct'].min())
                risk_data = {
                    'risk_score': min(100, max(0, price_volatility * 3)),
                    'volatility': min(100, max(0, price_volatility * 2)),
                    'performance': min(100, max(0, avg_performance + 50)),
                    'sentiment': 50 + (avg_performance * 0.8),
                    'liquidity': min(100, max(20, 80 - abs(max_loss)))
                }
                gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                with gauge_col1:
                    risk_fig = risk_gauge.create_risk_gauge(risk_data['risk_score'], "Market Risk")
                    st.plotly_chart(risk_fig, use_container_width=True)
                with gauge_col2:
                    vol_fig = risk_gauge.create_volatility_gauge(risk_data['volatility'])
                    st.plotly_chart(vol_fig, use_container_width=True)
                with gauge_col3:
                    perf_fig = risk_gauge.create_performance_gauge(avg_performance)
                    st.plotly_chart(perf_fig, use_container_width=True)
        except Exception as e:
            st.info("Risk assessment calculations temporarily unavailable.")
        st.subheader("📊 Detailed Stock Analysis")
        all_stocks_df = merged_data.copy()
        if not all_stocks_df.empty and 'Price_Change_Pct' in all_stocks_df.columns:
            valid_data = all_stocks_df[all_stocks_df['Price_Change_Pct'].notna()].copy()
            if not valid_data.empty:
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                with filter_col1:
                    available_symbols = comp_analysis.get_all_symbols()
                    if available_symbols:
                        search_symbol = st.selectbox("Select Stock Symbol:", ["None"] + available_symbols, index=0, help="Select a symbol to analyze", key="filter_search_symbol")
                        if search_symbol != "None":
                            st.session_state.selected_stock_symbol = search_symbol
                    else:
                        search_symbol = None
                        st.info("No symbols available")
                with filter_col2:
                    available_sectors = comp_analysis.get_available_sectors()
                    if available_sectors:
                        selected_sector = st.selectbox("Filter by Sector:", available_sectors, index=0, key="filter_sector")
                    else:
                        selected_sector = "All"
                        st.info("Sector data not available")
                with filter_col3:
                    performance_filter = st.selectbox(
                        "Filter by Performance",
                        ["All Stocks", "Gainers Only", "Losers Only", "Top 10 Performers", "Bottom 10 Performers"],
                        help="Filter stocks based on performance",
                        key="filter_performance"
                    )
                with filter_col4:
                    min_change = st.number_input("Min Change %", value=float(valid_data['Price_Change_Pct'].min()), help="Minimum price change percentage", key="filter_min_change")
                filter_col5, filter_col6 = st.columns(2)
                with filter_col5:
                    max_change = st.number_input("Max Change %", value=float(valid_data['Price_Change_Pct'].max()), help="Maximum price change percentage", key="filter_max_change")
                with filter_col6:
                    clear_filters_col1, clear_filters_col2 = st.columns(2)
                    with clear_filters_col1:
                        if st.button("🔍 Search & Filter", type="primary"):
                            st.success("Filters applied!")
                    with clear_filters_col2:
                        if st.button("🧹 Clear Filters", type="secondary", key="clear_filters_button"):
                            # Clear all filter-related keys
                            filter_keys = [key for key in st.session_state.keys() if 'filter_' in key or 'selected_stock_symbol' in key]
                            for key in filter_keys:
                                del st.session_state[key]
                            st.success("All filters cleared and reset to defaults!")
                            st.rerun()
                filter_col4, filter_col5 = st.columns(2)
                with filter_col4:
                    profit_filter = st.number_input("Min Profit $", value=0.0, help="Show only stocks with profit above this amount", key="filter_profit")
                with filter_col5:
                    loss_filter = st.number_input("Max Loss $", value=0.0, help="Show only stocks with loss below this amount (negative)", key="filter_loss")
                filtered_df = valid_data[(valid_data['Price_Change_Pct'] >= min_change) & (valid_data['Price_Change_Pct'] <= max_change)].copy()
                if search_symbol and search_symbol != "None":
                    filtered_df = filtered_df[filtered_df['Symbol'] == search_symbol]
                    st.info(f"Filtered to show results for: {search_symbol}")
                if selected_sector and selected_sector != "All":
                    sector_filtered = comp_analysis.filter_by_sector(selected_sector)
                    if not sector_filtered.empty:
                        sector_symbols = sector_filtered['Symbol'].tolist()
                        filtered_df = filtered_df[filtered_df['Symbol'].isin(sector_symbols)]
                        st.info(f"Filtered to show {selected_sector} sector stocks")
                if 'Price_Change' in filtered_df.columns:
                    if profit_filter > 0:
                        filtered_df = filtered_df[filtered_df['Price_Change'] >= profit_filter]
                    if loss_filter < 0:
                        filtered_df = filtered_df[filtered_df['Price_Change'] <= loss_filter]
                if performance_filter == "Gainers Only":
                    filtered_df = filtered_df[filtered_df['Price_Change_Pct'] > 0]
                elif performance_filter == "Losers Only":
                    filtered_df = filtered_df[filtered_df['Price_Change_Pct'] < 0]
                elif performance_filter == "Top 10 Performers":
                    if len(filtered_df) > 0 and 'Price_Change_Pct' in filtered_df.columns:
                        filtered_df = filtered_df.nlargest(10, 'Price_Change_Pct')
                elif performance_filter == "Bottom 10 Performers":
                    if len(filtered_df) > 0 and 'Price_Change_Pct' in filtered_df.columns:
                        filtered_df = filtered_df.nsmallest(10, 'Price_Change_Pct')
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    st.metric("Filtered Stocks", len(filtered_df))
                with metrics_col2:
                    gainers = len(filtered_df[filtered_df['Price_Change_Pct'] > 0]) if not filtered_df.empty else 0
                    st.metric("Gainers", gainers, delta="positive" if gainers > 0 else None)
                with metrics_col3:
                    losers = len(filtered_df[filtered_df['Price_Change_Pct'] < 0]) if not filtered_df.empty else 0
                    st.metric("Losers", losers, delta="negative" if losers > 0 else None)
                with metrics_col4:
                    avg_change = filtered_df['Price_Change_Pct'].mean() if not filtered_df.empty else 0
                    st.metric("Avg Change", f"{avg_change:.2f}%", delta=f"{avg_change:.2f}%")
                if not filtered_df.empty:
                    display_columns = []
                    basic_columns = ['Symbol']
                    for col in filtered_df.columns:
                        if 'Name_current' in col or 'Company_current' in col:
                            basic_columns.append(col)
                            break
                    for col in filtered_df.columns:
                        if any(x in col for x in ['Sector_current', 'Industry_current', 'Country_current', 'IPO_current']):
                            basic_columns.append(col)
                    display_columns.extend(basic_columns)
                    price_columns = []
                    for col in filtered_df.columns:
                        if 'Last Sale_previous' in col:
                            price_columns.append(col)
                        elif 'Last Sale_current' in col:
                            price_columns.append(col)
                    if 'Price_Change' in filtered_df.columns:
                        price_columns.append('Price_Change')
                    if 'Price_Change_Pct' in filtered_df.columns:
                        price_columns.append('Price_Change_Pct')
                    if 'Profit_Loss' in filtered_df.columns:
                        price_columns.append('Profit_Loss')
                    display_columns.extend(price_columns)
                    mcap_columns = []
                    for col in filtered_df.columns:
                        if 'Market Cap_previous' in col:
                            mcap_columns.append(col)
                        elif 'Market Cap_current' in col:
                            mcap_columns.append(col)
                    if 'MarketCap_Difference' in filtered_df.columns:
                        mcap_columns.append('MarketCap_Difference')
                    display_columns.extend(mcap_columns)
                    volume_columns = []
                    for col in filtered_df.columns:
                        if 'Volume_previous' in col:
                            volume_columns.append(col)
                        elif 'Volume_current' in col:
                            volume_columns.append(col)
                    if 'Volume_Difference' in filtered_df.columns:
                        volume_columns.append('Volume_Difference')
                    display_columns.extend(volume_columns)
                    available_columns = [col for col in display_columns if col in filtered_df.columns]
                    if available_columns:
                        display_data = filtered_df[available_columns].copy()
                        for col in display_data.columns:
                            if display_data[col].dtype in ['float64', 'int64']:
                                display_data[col] = display_data[col].round(2)
                        st.dataframe(display_data, use_container_width=True, height=400)
                        st.info(f"Displaying {len(available_columns)} columns: {', '.join(available_columns[:5])}{'...' if len(available_columns) > 5 else ''}")
                    else:
                        st.warning("No comparison columns available for display.")
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data",
                        data=csv,
                        file_name=f"filtered_stocks_{performance_filter.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No stocks match the current filter criteria")
            else:
                st.warning("No valid price change data available for analysis")
        else:
            st.warning("Price change analysis not available - missing required data")
        st.subheader("🏭 Sector Performance Analysis")
        try:
            sector_chart = comp_analysis.create_sector_performance_chart()
            if sector_chart and hasattr(sector_chart, 'data') and sector_chart.data:
                st.plotly_chart(sector_chart, use_container_width=True)
            else:
                st.info("Sector performance chart not available")
        except Exception as e:
            st.warning(f"Could not generate sector chart: {str(e)}")
        st.subheader("🏭 Detailed Sector Analysis")
        try:
            sector_analysis = comp_analysis.get_sector_analysis()
            if not sector_analysis.empty:
                st.dataframe(sector_analysis, use_container_width=True)
            else:
                st.info("Sector analysis not available - missing sector data")
        except Exception as e:
            st.warning(f"Could not generate sector analysis: {str(e)}")
        st.subheader("🏢 Industry Analysis (Top 20)")
        try:
            industry_analysis = comp_analysis.get_industry_analysis()
            if not industry_analysis.empty:
                st.dataframe(industry_analysis, use_container_width=True)
            else:
                st.info("Industry analysis not available - missing industry data")
        except Exception as e:
            st.warning(f"Could not generate industry analysis: {str(e)}")
        st.subheader("🔗 Correlation Analysis")
        try:
            correlation_chart = comp_analysis.create_correlation_heatmap()
            if correlation_chart and hasattr(correlation_chart, 'data') and correlation_chart.data:
                st.plotly_chart(correlation_chart, use_container_width=True)
            else:
                st.info("Correlation analysis not available")
        except Exception as e:
            st.warning(f"Could not generate correlation analysis: {str(e)}")
        st.subheader("📊 Comprehensive Performance Dashboard")
        try:
            dashboard_fig = comp_analysis.create_performance_dashboard()
            if dashboard_fig and hasattr(dashboard_fig, 'data') and dashboard_fig.data:
                st.plotly_chart(dashboard_fig, use_container_width=True)
            else:
                st.info("Performance dashboard not available")
        except Exception as e:
            st.warning(f"Could not generate dashboard: {str(e)}")
        with st.expander("📊 Per-Stock Statistical Summary", expanded=False):
            if 'Price_Change_Pct' in merged_data.columns:
                st.subheader("Individual Stock Statistics")
                stock_stats = merged_data[['Symbol', 'Current_Price', 'Previous_Price', 'Price_Change', 'Price_Change_Pct']].copy()
                stock_stats = stock_stats.sort_values('Price_Change_Pct', ascending=False)
                st.dataframe(
                    stock_stats.style.format({
                        'Current_Price': '${:.2f}',
                        'Previous_Price': '${:.2f}',
                        'Price_Change': '${:.2f}',
                        'Price_Change_Pct': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                st.subheader("Aggregate Statistics")
                stats_data = merged_data['Price_Change_Pct'].describe()
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Mean Change", f"{stats_data['mean']:.2f}%")
                    st.metric("Median Change", f"{stats_data['50%']:.2f}%")
                    st.metric("Standard Deviation", f"{stats_data['std']:.2f}%")
                with stats_col2:
                    st.metric("Maximum Gain", f"{stats_data['max']:.2f}%")
                    st.metric("Maximum Loss", f"{stats_data['min']:.2f}%")
                    st.metric("Data Points", f"{int(stats_data['count'])}")
            else:
                st.info("Price change statistics not available")
    except Exception as e:
        st.error(f"Error in Phase 1 analysis: {str(e)}")

def phase2_deep_analysis_section():
    st.header("📈 Phase 2: Deep Stock Analysis")
    st.markdown("Comprehensive technical analysis with custom date ranges and yfinance integration")
    available_stocks = []
    if st.session_state.current_data is not None and 'Symbol' in st.session_state.current_data.columns:
        available_stocks.extend(st.session_state.current_data['Symbol'].dropna().unique().tolist())
    if st.session_state.previous_data is not None and 'Symbol' in st.session_state.previous_data.columns:
        prev_stocks = st.session_state.previous_data['Symbol'].dropna().unique().tolist()
        available_stocks.extend([s for s in prev_stocks if s not in available_stocks])
    if not available_stocks:
        st.warning("⚠️ Please upload stock data in the Data Upload tab first.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Stock Selection")
        suggestions = []
        if st.session_state.comparative_analysis is not None:
            try:
                comp_data = st.session_state.comparative_analysis.merged_data
                if 'Price_Change_Pct' in comp_data.columns:
                    top_performers = comp_data.nlargest(3, 'Price_Change_Pct')['Symbol'].tolist()
                    suggestions = [f"🏆 {stock} (Top Performer)" for stock in top_performers]
            except:
                pass
        if suggestions:
            st.info("💡 Suggested stocks from Phase 1 analysis:")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        selected_stock = st.selectbox("Select Stock for Deep Analysis", options=available_stocks, help="Choose a stock from your uploaded data for comprehensive analysis")
        if selected_stock:
            st.session_state.selected_stock_symbol = selected_stock
            st.success(f"✅ {selected_stock} selected for analysis and will be used in Advanced Analytics")
    with col2:
        st.subheader("📅 Time Period Selection")
        date_option = st.radio("Choose date range option:", ["Predefined Periods", "Custom Date Range"])
        if date_option == "Predefined Periods":
            analysis_period = st.selectbox("Analysis Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3, help="Predefined time periods for analysis")
            start_date, end_date = None, None
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365), help="Select the start date for analysis")
            with col_end:
                end_date = st.date_input("End Date", value=datetime.now(), help="Select the end date for analysis")
            analysis_period = None
    if selected_stock:
        st.markdown("---")
        if st.button("🔄 Fetch & Analyze Data", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Fetching comprehensive data for {selected_stock}..."):
                    ticker = yf.Ticker(selected_stock)
                    if analysis_period:
                        hist_data = ticker.history(period=analysis_period)
                    else:
                        hist_data = ticker.history(start=start_date, end=end_date)
                    if hist_data.empty:
                        st.error(f"No data available for {selected_stock} from yfinance for the selected period")
                        return
                    # Clean yfinance data to avoid Datetime ambiguity
                    hist_data = hist_data.reset_index()
                    hist_data = hist_data.rename(columns={'Date': 'Datetime'})
                    hist_data['Datetime'] = pd.to_datetime(hist_data['Datetime'])
                    st.session_state.yfinance_data = hist_data
                    st.session_state.selected_symbol = selected_stock
                    try:
                        info = ticker.info
                        company_name = info.get('longName', selected_stock)
                        sector = info.get('sector', 'Unknown')
                        industry = info.get('industry', 'Unknown')
                        market_cap = info.get('marketCap', 'Unknown')
                    except:
                        company_name = selected_stock
                        sector = 'Unknown'
                        industry = 'Unknown'
                        market_cap = 'Unknown'
                if st.session_state.yfinance_data is not None and not st.session_state.yfinance_data.empty:
                    hist_data = st.session_state.yfinance_data
                    start_date = pd.to_datetime(hist_data['Datetime'].iloc[0]).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(hist_data['Datetime'].iloc[-1]).strftime('%Y-%m-%d')
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>📋 {company_name} ({selected_stock})</h3>
                        <p>Data Period: {start_date} to {end_date} ({len(hist_data)} trading days)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        current_price = hist_data['Close'].iloc[-1]
                        latest_date = pd.to_datetime(hist_data['Datetime'].iloc[-1]).strftime('%m-%d-%y')
                        st.metric("Current Price", f"${current_price:.2f}", help=f"As of {latest_date}")
                    with col2:
                        if len(hist_data) > 1:
                            price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
                            price_change_pct = (price_change / hist_data['Close'].iloc[-2]) * 100
                            previous_date = pd.to_datetime(hist_data['Datetime'].iloc[-2]).strftime('%m-%d-%y')
                            st.metric("Daily Change", f"${price_change:.2f}", delta=f"{price_change_pct:.2f}%")
                        else:
                            st.metric("Daily Change", "N/A")
                    with col3:
                        st.metric("Sector", sector)
                    with col4:
                        st.metric("Industry", industry)
                    with col5:
                        if isinstance(market_cap, (int, float)):
                            market_cap_b = market_cap / 1e9
                            st.metric("Market Cap", f"${market_cap_b:.1f}B")
                        else:
                            st.metric("Market Cap", str(market_cap))
                    hist_data_clean = hist_data.copy()
                    if 'Adj Close' not in hist_data_clean.columns:
                        hist_data_clean['Adj Close'] = hist_data_clean['Close']
                        st.info("ℹ️ Using Close price for analysis (Adj Close not available)")
                    for col in ['Dividends', 'Stock Splits']:
                        if col not in hist_data_clean.columns:
                            hist_data_clean[col] = 0
                    st.subheader("📊 Advanced Price Visualizations")
                    if len(hist_data_clean) > 0:
                        viz = Visualizations(historical_data=hist_data_clean)
                        candlestick_fig = viz.create_candlestick_chart()
                        st.plotly_chart(candlestick_fig, use_container_width=True, key="phase2_candlestick")
                        price_trends_fig = viz.create_price_trends_chart()
                        st.plotly_chart(price_trends_fig, use_container_width=True, key="phase2_price_trends")
                        volume_fig = viz.create_volume_chart()
                        st.plotly_chart(volume_fig, use_container_width=True, key="phase2_volume")
                    if len(hist_data_clean) > 50:
                        st.subheader("⚙️ Technical Indicators Dashboard")
                        tech_indicators = TechnicalIndicators(hist_data_clean)
                        ma_chart = tech_indicators.create_moving_averages_chart()
                        st.plotly_chart(ma_chart, use_container_width=True, key="phase2_ma_chart")
                        tech_col1, tech_col2 = st.columns(2)
                        with tech_col1:
                            rsi_chart = tech_indicators.create_rsi_chart()
                            st.plotly_chart(rsi_chart, use_container_width=True, key="phase2_rsi_chart")
                        with tech_col2:
                            macd_chart = tech_indicators.create_macd_chart()
                            st.plotly_chart(macd_chart, use_container_width=True, key="phase2_macd_chart")
                        bb_chart = tech_indicators.create_bollinger_bands_chart()
                        st.plotly_chart(bb_chart, use_container_width=True, key="phase2_bb_chart")
                        st.subheader("🎯 Trading Signals & Recommendations")
                        signals = tech_indicators.get_trading_signals()
                        signal_cols = st.columns(min(len(signals), 4))
                        for i, (indicator, signal_data) in enumerate(signals.items()):
                            with signal_cols[i % len(signal_cols)]:
                                signal_value = signal_data.get('signal', 'Unknown')
                                # Placeholder strength if not provided
                                strength = signal_data.get('strength', 'Moderate (Pending Implementation)')
                                if 'buy' in signal_value.lower():
                                    st.markdown(f"""
                                    <div class="success-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif 'sell' in signal_value.lower():
                                    st.markdown(f"""
                                    <div class="warning-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    st.subheader("📈 Performance Metrics")
                    if len(hist_data) > 20:
                        returns = hist_data['Close'].pct_change().dropna()
                        perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
                        with perf_col1:
                            total_return = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1) * 100
                            st.metric("Total Return", f"{total_return:.2f}%")
                        with perf_col2:
                            volatility = returns.std() * np.sqrt(252) * 100
                            st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                        with perf_col3:
                            max_price = hist_data['Close'].max()
                            current_price = hist_data['Close'].iloc[-1]
                            drawdown = ((current_price - max_price) / max_price) * 100
                            st.metric("Current Drawdown", f"{drawdown:.2f}%")
                        with perf_col4:
                            if len(returns) > 0 and returns.std() > 0:
                                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                        with perf_col5:
                            avg_volume = hist_data['Volume'].mean()
                            st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    if st.session_state.comparative_analysis is not None:
                        st.subheader("🔄 Phase 1 Integration")
                        comp_data = st.session_state.comparative_analysis.merged_data
                        stock_comp_data = comp_data[comp_data['Symbol'] == selected_stock]
                        if not stock_comp_data.empty:
                            stock_row = stock_comp_data.iloc[0]
                            comp_col1, comp_col2, comp_col3 = st.columns(3)
                            with comp_col1:
                                if 'Price_Change_Pct' in stock_row:
                                    st.metric("Period Change (Phase 1)", f"{stock_row['Price_Change_Pct']:.2f}%")
                            with comp_col2:
                                if 'Sector_current' in stock_row:
                                    sector_avg = comp_data[comp_data['Sector_current'] == stock_row['Sector_current']]['Price_Change_Pct'].mean()
                                    st.metric("Sector Average", f"{sector_avg:.2f}%")
                            with comp_col3:
                                if 'Industry_current' in stock_row:
                                    industry_avg = comp_data[comp_data['Industry_current'] == stock_row['Industry_current']]['Price_Change_Pct'].mean()
                                    st.metric("Industry Average", f"{industry_avg:.2f}%")
                        else:
                            st.info("Stock not found in Phase 1 comparative analysis data")
            except Exception as e:
                st.error(f"Error fetching data for {selected_stock}: {str(e)}")
                st.info("Please check if the stock symbol is valid and try again.")

def advanced_analytics_section():
    st.header("🔮 Advanced Analytics & Predictions")
    st.markdown("Advanced technical analysis, predictions, and comprehensive trading insights")
    if st.session_state.selected_stock_symbol:
        st.success(f"📊 Analyzing: {st.session_state.selected_stock_symbol} (selected from Phase 2)")
    if st.session_state.historical_data is None and st.session_state.yfinance_data is None:
        st.warning("⚠️ Please upload historical data or fetch yfinance data in Phase 2 first.")
        return
    data_source = st.session_state.yfinance_data if st.session_state.yfinance_data is not None else st.session_state.historical_data
    if data_source is None or data_source.empty:
        st.error("No historical data available for analysis.")
        return
    # Clean data to avoid Datetime ambiguity
    data_clean = data_source.copy()
    if hasattr(data_clean, 'reset_index'):
        data_clean = data_clean.reset_index()
    if 'Datetime' in data_clean.columns and data_clean.index.name == 'Datetime':
        data_clean = data_clean.drop(columns=['Datetime'])
    elif 'Date' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Date': 'Datetime'})
    if 'Datetime' not in data_clean.columns:
        data_clean['Datetime'] = data_clean.index
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    if 'Adj Close' not in data_clean.columns:
        data_clean['Adj Close'] = data_clean['Close']
    for col in ['Dividends', 'Stock Splits']:
        if col not in data_clean.columns:
            data_clean[col] = 0
    pred_tab, viz_tab, insights_tab = st.tabs(["🔮 Price Predictions", "📊 Advanced Visualizations", "💡 Trading Insights"])
    with pred_tab:
        st.subheader("🔮 Price Predictions")
        if len(data_clean) > 50:
            predictions = PricePredictions(data_clean)
            col1, col2 = st.columns(2)
            with col1:
                pred_days = st.slider("Prediction Days", min_value=1, max_value=30, value=7)
            with col2:
                pred_method = st.selectbox(
                    "Prediction Method",
                    ["technical_analysis", "linear_trend", "moving_average"],
                    format_func=lambda x: {"technical_analysis": "Technical Analysis", "linear_trend": "Linear Trend", "moving_average": "Moving Average"}[x]
                )
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating price predictions..."):
                    pred_prices = predictions.predict_prices(pred_days, pred_method)
                    if pred_prices:
                        pred_chart = predictions.create_prediction_chart(pred_prices, pred_days)
                        st.plotly_chart(pred_chart, use_container_width=True, key="predictions_chart")
                        st.subheader("📈 Predicted Prices")
                        from datetime import datetime, timedelta
                        current_date = datetime.now()
                        pred_dates = [current_date + timedelta(days=i+1) for i in range(pred_days)]
                        pred_df = pd.DataFrame({
                            'Date': [d.strftime('%Y-%m-%d') for d in pred_dates],
                            'Day': [f"Day {i+1}" for i in range(pred_days)],
                            'Predicted Price': [f"${price:.2f}" for price in pred_prices]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            current_price = data_clean['Close'].iloc[-1]
                            predicted_final = pred_prices[-1]
                            change_pct = ((predicted_final - current_price) / current_price) * 100
                            st.metric("Predicted Change", f"{change_pct:.2f}%")
                        with col2:
                            st.metric("Target Price", f"${predicted_final:.2f}")
                        with col3:
                            confidence = predictions.calculate_prediction_confidence()
                            st.metric("Confidence Score", f"{confidence.get('score', 0):.1f}/10")
                        with col4:
                            volatility = confidence.get('volatility', 0)
                            st.metric("Prediction Volatility", f"{volatility:.2f}%")
                        st.info(predictions.get_prediction_disclaimer())
                    else:
                        st.error("Unable to generate predictions. Please try a different method.")
        else:
            st.warning("Insufficient data for predictions (need >50 data points)")
    with viz_tab:
        st.subheader("📊 Advanced Visualizations")
        with st.expander("📚 Understanding 3D Factor Analysis", expanded=False):
            st.markdown("""
            **3D Factor Analysis** is an advanced visualization technique that simultaneously displays three critical financial dimensions:
            **The Three Dimensions Explained:**
            - **X-Axis (Risk Level):** Volatility measure - how much prices fluctuate
            - **Y-Axis (Return Potential):** Expected gains based on historical performance
            - **Z-Axis (Market Correlation):** How closely the stock follows overall market trends
            - **Color Gradient:** Performance score indicating overall investment success
            **How to Interpret the 3D Surface:**
            - **High Peaks:** Represent optimal investment opportunities with strong performance across all factors
            - **Valleys:** Indicate potential risks or underperforming areas
            - **Clusters:** Groups of similar investment characteristics
            - **Outliers:** Unique opportunities requiring special attention
            **Real-time Example:**
            If you see a bright peak at coordinates (30, 80, 60), this represents:
            - Medium risk level (30% volatility)
            - High return potential (80% expected gains)
            - Moderate market correlation (60% follows market trends)
            - This would be ideal for growth-oriented investors seeking balanced risk-return profiles
            **Practical Use Cases:**
            - **Portfolio Diversification:** Find stocks with different correlation patterns
            - **Risk Management:** Identify high-return, low-risk opportunities
            - **Market Timing:** Understand how stocks react to market movements
            - **Investment Strategy:** Match investments to your risk tolerance and return expectations
            """)
        if len(data_clean) > 0:
            viz = Visualizations(historical_data=data_clean)
            risk_gauge = RiskGauge()
            st.subheader("📈 Candlestick Chart")
            try:
                candlestick_fig = viz.create_candlestick_chart()
                st.plotly_chart(candlestick_fig, use_container_width=True, key="advanced_candlestick")
            except Exception as e:
                st.warning(f"Could not generate Candlestick Chart: {str(e)}")
            st.subheader("📉 Price Trends")
            try:
                price_trends_fig = viz.create_price_trends_chart()
                st.plotly_chart(price_trends_fig, use_container_width=True, key="advanced_price_trends")
            except Exception as e:
                st.warning(f"Could not generate Price Trends Chart: {str(e)}")
            st.subheader("📊 Volume Analysis")
            try:
                volume_fig = viz.create_volume_chart()
                st.plotly_chart(volume_fig, use_container_width=True, key="advanced_volume")
            except Exception as e:
                st.warning(f"Could not generate Volume Analysis Chart: {str(e)}")
            st.subheader("🌐 3D Factor Analysis")
            if len(data_clean) > 20:
                try:
                    returns = data_clean['Close'].pct_change().dropna()
                    volatility = returns.std() * 100
                    total_return = ((data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0]) - 1) * 100
                    risk_levels = np.linspace(0, 100, 20)
                    return_levels = np.linspace(-50, 150, 20)
                    X, Y = np.meshgrid(risk_levels, return_levels)
                    Z = 100 * np.exp(-((X-volatility)**2 + (Y-total_return)**2) / 2000)
                    fig = go.Figure(data=[go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Performance Score")
                    )])
                    fig.update_layout(
                        title='3D Factor Analysis: Risk vs Return vs Performance',
                        scene=dict(
                            xaxis_title='Risk Level (%)',
                            yaxis_title='Return Potential (%)',
                            zaxis_title='Performance Score',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True, key="3d_factor_analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Risk Level", f"{volatility:.1f}%")
                    with col2:
                        st.metric("Total Return", f"{total_return:.1f}%")
                    with col3:
                        performance_score = min(100, max(0, 50 + total_return/2))
                        st.metric("Performance Score", f"{performance_score:.1f}/100")
                except Exception as e:
                    st.warning(f"Could not generate 3D Factor Analysis: {str(e)}")
            else:
                st.warning("Need more data points for 3D Factor Analysis (minimum 20 days)")
            st.subheader("🌐 3D Risk Surface Analysis")
            if len(data_clean) >= 10:
                try:
                    surface_fig = risk_gauge.create_3d_surface_plot(data_clean)
                    st.plotly_chart(surface_fig, use_container_width=True, key="3d_risk_surface")
                except Exception as e:
                    st.warning(f"Could not generate 3D Risk Surface Analysis: {str(e)}")
            else:
                st.warning("Need at least 10 data points for 3D Risk Surface Analysis")
            if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                st.subheader("📊 Market Overview Dashboard")
                try:
                    daily_viz = Visualizations(daily_data=st.session_state.current_data)
                    market_overview_fig = daily_viz.create_market_overview_dashboard()
                    st.plotly_chart(market_overview_fig, use_container_width=True, key="market_overview_dashboard")
                except Exception as e:
                    st.warning(f"Could not generate Market Overview Dashboard: {str(e)}")
            else:
                st.warning("Current stock data required for Market Overview Dashboard")
            if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                st.subheader("📈 Market Analysis Visualizations")
                daily_viz = Visualizations(daily_data=st.session_state.current_data)
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    try:
                        market_cap_fig = daily_viz.create_market_cap_chart()
                        st.plotly_chart(market_cap_fig, use_container_width=True, key="advanced_market_cap")
                    except Exception as e:
                        st.warning(f"Could not generate Market Cap Chart: {str(e)}")
                with viz_col2:
                    try:
                        sector_fig = daily_viz.create_sector_pie_chart()
                        st.plotly_chart(sector_fig, use_container_width=True, key="advanced_sector_pie")
                    except Exception as e:
                        st.warning(f"Could not generate Sector Pie Chart: {str(e)}")
                try:
                    corr_fig = daily_viz.create_correlation_heatmap()
                    st.plotly_chart(corr_fig, use_container_width=True, key="advanced_correlation")
                except Exception as e:
                    st.warning(f"Could not generate Correlation Heatmap: {str(e)}")
                try:
                    perf_fig = daily_viz.create_performance_volume_scatter()
                    st.plotly_chart(perf_fig, use_container_width=True, key="advanced_performance_scatter")
                except Exception as e:
                    st.warning(f"Could not generate Performance Scatter: {str(e)}")
            st.subheader("🎯 Professional Risk Assessment Dashboard")
            if len(data_clean) > 1:
                try:
                    price_volatility = data_clean['Close'].pct_change().std() * 100
                    volume_volatility = data_clean['Volume'].pct_change().std() * 100 if 'Volume' in data_clean.columns else 30
                    price_trend = (data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0] - 1) * 100
                    risk_data = {
                        'risk_score': min(100, max(0, price_volatility * 2)),
                        'volatility': min(100, price_volatility),
                        'performance': min(100, max(0, price_trend + 50)),
                        'sentiment': 50 + (price_trend * 0.5),
                        'liquidity': min(100, max(20, 100 - volume_volatility))
                    }
                    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                    with gauge_col1:
                        risk_fig = risk_gauge.create_risk_gauge(risk_data['risk_score'], "Overall Risk")
                        st.plotly_chart(risk_fig, use_container_width=True, key="risk_gauge")
                    with gauge_col2:
                        vol_fig = risk_gauge.create_volatility_gauge(risk_data['volatility'])
                        st.plotly_chart(vol_fig, use_container_width=True, key="volatility_gauge")
                    with gauge_col3:
                        perf_fig = risk_gauge.create_performance_gauge(price_trend)
                        st.plotly_chart(perf_fig, use_container_width=True, key="performance_gauge")
                    st.subheader("📊 Advanced Multi-Metric Dashboard")
                    advanced_fig = risk_gauge.create_advanced_dashboard(risk_data)
                    st.plotly_chart(advanced_fig, use_container_width=True, key="advanced_dashboard")
                except Exception as e:
                    st.warning(f"Could not generate Risk Assessment Dashboard: {str(e)}")
            else:
                st.warning("Insufficient data for risk assessment")
            st.subheader("🔥 Advanced Correlation Matrix")
            try:
                heatmap_fig = risk_gauge.create_heatmap_correlation(data_clean)
                st.plotly_chart(heatmap_fig, use_container_width=True, key="correlation_heatmap")
            except Exception as e:
                st.warning(f"Could not generate Correlation Matrix: {str(e)}")
            if 'Date' not in data_clean.columns:
                data_clean = data_clean.reset_index()
                if 'Date' not in data_clean.columns:
                    data_clean['Date'] = data_clean.index
            st.subheader("📈 Professional Technical Analysis Chart")
            try:
                candlestick_fig = risk_gauge.create_advanced_candlestick(data_clean)
                st.plotly_chart(candlestick_fig, use_container_width=True, key="advanced_candlestick_tech")
            except Exception as e:
                st.warning(f"Could not generate Advanced Candlestick Chart: {str(e)}")
    with insights_tab:
        st.subheader("💡 Enhanced Trading Insights")
        if len(data_clean) > 50:
            tech_indicators = TechnicalIndicators(data_clean)
            trading_signals = tech_indicators.get_trading_signals()
            analytics = Analytics(historical_data=data_clean)
            st.subheader("🎯 Current Trading Signals")
            signal_summary = {"buy": 0, "sell": 0, "hold": 0}
            for indicator, signal_data in trading_signals.items():
                signal = signal_data.get('signal', '').lower()
                if 'buy' in signal:
                    signal_summary["buy"] += 1
                elif 'sell' in signal:
                    signal_summary["sell"] += 1
                else:
                    signal_summary["hold"] += 1
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Buy Signals", signal_summary["buy"])
            with col2:
                st.metric("🔴 Sell Signals", signal_summary["sell"])
            with col3:
                st.metric("🟡 Hold/Neutral", signal_summary["hold"])
            st.subheader("📋 Detailed Signal Analysis")
            for indicator, signal_data in trading_signals.items():
                signal_value = signal_data.get('signal', 'Unknown')
                strength = signal_data.get('strength', 'Moderate (Pending Implementation)')
                with st.expander(f"{indicator} - {signal_value}"):
                    st.write(f"**Current Signal:** {signal_value}")
                    st.write(f"**Signal Strength:** {strength}")
                    if indicator == "RSI":
                        rsi_val = tech_indicators.calculate_rsi().iloc[-1] if len(tech_indicators.calculate_rsi()) > 0 else 0
                        st.write(f"**Current RSI:** {rsi_val:.1f}")
                        if rsi_val > 70:
                            st.warning("Stock may be overbought - consider taking profits")
                        elif rsi_val < 30:
                            st.success("Stock may be oversold - potential buying opportunity")
                    elif indicator == "MACD":
                        st.write("Monitor MACD line crossovers for momentum changes")
                    elif indicator == "Bollinger Bands":
                        bb_position = tech_indicators.get_bollinger_position()
                        st.write(f"**Current Position:** {bb_position}")
            st.subheader("🎯 Strategy Recommendations")
            strategies = analytics.generate_trading_strategies(trading_signals)
            if strategies:
                for i, strategy in enumerate(strategies):
                    with st.expander(f"Strategy {i+1}: {strategy.get('name', 'Unknown')}"):
                        st.write(f"**Type:** {strategy.get('type', 'N/A')}")
                        st.write(f"**Risk Level:** {strategy.get('risk_level', 'N/A')}")
                        st.write(f"**Time Horizon:** {strategy.get('time_horizon', 'N/A')}")
                        st.write(f"**Description:** {strategy.get('description', 'N/A')}")
                        if 'entry_conditions' in strategy:
                            st.write(f"**Entry Conditions:** {strategy['entry_conditions']}")
                        if 'exit_conditions' in strategy:
                            st.write(f"**Exit Conditions:** {strategy['exit_conditions']}")
                        if 'risk_management' in strategy:
                            st.write(f"**Risk Management:** {strategy['risk_management']}")
            st.subheader("⚠️ Risk Assessment")
            risk_metrics = analytics.calculate_risk_metrics()
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            with risk_col1:
                st.metric("Beta (Market Risk)", f"{risk_metrics.get('beta', 'N/A')}")
            with risk_col2:
                st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 'N/A')}")
            with risk_col3:
                st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 'N/A')}%")
            with risk_col4:
                st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 'N/A')}%")
            st.subheader("📈 Market Patterns Analysis")
            patterns = analytics.analyze_patterns()
            pattern_col1, pattern_col2 = st.columns(2)
            with pattern_col1:
                st.write("**Seasonal Patterns**")
                if patterns.get('seasonal_patterns'):
                    for pattern in patterns['seasonal_patterns']:
                        st.write(f"• {pattern}")
                else:
                    st.write("No significant seasonal patterns detected.")
            with pattern_col2:
                st.write("**Volume Patterns**")
                if patterns.get('volume_patterns'):
                    for pattern in patterns['volume_patterns']:
                        st.write(f"• {pattern}")
                else:
                    st.write("No significant volume patterns detected.")
            st.subheader("🎯 Overall Trading Recommendation")
            buy_signals = signal_summary["buy"]
            sell_signals = signal_summary["sell"]
            if buy_signals > sell_signals:
                st.success(f"**BULLISH OUTLOOK** - {buy_signals} buy signals vs {sell_signals} sell signals")
                st.write("Consider position accumulation or holding existing positions")
            elif sell_signals > buy_signals:
                st.error(f"**BEARISH OUTLOOK** - {sell_signals} sell signals vs {buy_signals} buy signals")
                st.write("Consider reducing positions or implementing defensive strategies")
            else:
                st.info(f"**NEUTRAL OUTLOOK** - Mixed signals ({buy_signals} buy, {sell_signals} sell)")
                st.write("Wait for clearer directional signals before taking major positions")
        else:
            st.warning("Insufficient data for comprehensive trading insights")
    if len(data_clean) > 0:
        st.markdown("---")
        st.subheader("📄 Download Comprehensive Report")
        st.info("Generate an interactive HTML report with all charts and hover functionality preserved")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            if st.button("🔽 Generate HTML Report", type="primary"):
                try:
                    from utils.html_report_generator import HTMLReportGenerator
                    stock_symbol = st.session_state.selected_stock_symbol or "STOCK"
                    # Prepare clean data for report
                    clean_data_for_components = data_clean.copy()
                    clean_data_for_components = clean_data_for_components.reset_index(drop=True)
                    if 'Datetime' in clean_data_for_components.columns:
                        clean_data_for_components['Datetime'] = pd.to_datetime(clean_data_for_components['Datetime'])
                    else:
                        clean_data_for_components['Datetime'] = pd.date_range(start='2020-01-01', periods=len(clean_data_for_components), freq='D')
                    tech_indicators = TechnicalIndicators(clean_data_for_components)
                    analytics = Analytics(historical_data=clean_data_for_components)
                    viz = Visualizations(historical_data=clean_data_for_components)
                    risk_gauge = RiskGauge()
                    visualizations_dict = {}
                    try:
                        visualizations_dict['candlestick'] = viz.create_candlestick_chart()
                        visualizations_dict['price_trends'] = viz.create_price_trends_chart()
                        visualizations_dict['volume_analysis'] = viz.create_volume_chart()
                        visualizations_dict['advanced_candlestick'] = risk_gauge.create_advanced_candlestick(clean_data_for_components)
                        visualizations_dict['correlation_heatmap'] = risk_gauge.create_heatmap_correlation(clean_data_for_components)
                    except Exception as e:
                        st.warning(f"Error generating standard visualizations for report: {str(e)}")
                    if len(clean_data_for_components) > 20:
                        try:
                            returns = clean_data_for_components['Close'].pct_change().dropna()
                            volatility = returns.std() * 100
                            total_return = ((clean_data_for_components['Close'].iloc[-1] / clean_data_for_components['Close'].iloc[0]) - 1) * 100
                            risk_levels = np.linspace(0, 100, 20)
                            return_levels = np.linspace(-50, 150, 20)
                            X, Y = np.meshgrid(risk_levels, return_levels)
                            Z = 100 * np.exp(-((X-volatility)**2 + (Y-total_return)**2) / 2000)
                            factor_fig = go.Figure(data=[go.Surface(
                                x=X, y=Y, z=Z,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Performance Score")
                            )])
                            factor_fig.update_layout(
                                title='3D Factor Analysis: Risk vs Return vs Performance',
                                scene=dict(
                                    xaxis_title='Risk Level (%)',
                                    yaxis_title='Return Potential (%)',
                                    zaxis_title='Performance Score',
                                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                                ),
                                height=600
                            )
                            visualizations_dict['3d_factor_analysis'] = factor_fig
                        except Exception as e:
                            st.warning(f"Error generating 3D Factor Analysis for report: {str(e)}")
                    else:
                        st.warning("Insufficient data for 3D Factor Analysis in report (minimum 20 days)")
                    if len(clean_data_for_components) >= 10:
                        try:
                            surface_fig = risk_gauge.create_3d_surface_plot(clean_data_for_components)
                            visualizations_dict['3d_risk_surface'] = surface_fig
                        except Exception as e:
                            st.warning(f"Error generating 3D Risk Surface Analysis for report: {str(e)}")
                    else:
                        st.warning("Insufficient data for 3D Risk Surface Analysis in report (minimum 10 days)")
                    if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                        try:
                            daily_viz = Visualizations(daily_data=st.session_state.current_data)
                            market_overview_fig = daily_viz.create_market_overview_dashboard()
                            visualizations_dict['market_overview_dashboard'] = market_overview_fig
                        except Exception as e:
                            st.warning(f"Error generating Market Overview Dashboard for report: {str(e)}")
                    else:
                        st.warning("Current stock data required for Market Overview Dashboard in report")
                    if st.session_state.current_data is not None and not st.session_state.current_data.empty:
                        daily_viz = Visualizations(daily_data=st.session_state.current_data)
                        try:
                            visualizations_dict['market_cap_chart'] = daily_viz.create_market_cap_chart()
                            visualizations_dict['sector_pie_chart'] = daily_viz.create_sector_pie_chart()
                            visualizations_dict['correlation_heatmap_daily'] = daily_viz.create_correlation_heatmap()
                            visualizations_dict['performance_scatter'] = daily_viz.create_performance_volume_scatter()
                        except Exception as e:
                            st.warning(f"Error generating daily visualizations for report: {str(e)}")
                    if len(clean_data_for_components) > 1:
                        try:
                            price_volatility = clean_data_for_components['Close'].pct_change().std() * 100
                            volume_volatility = clean_data_for_components['Volume'].pct_change().std() * 100 if 'Volume' in clean_data_for_components.columns else 30
                            price_trend = (clean_data_for_components['Close'].iloc[-1] / clean_data_for_components['Close'].iloc[0] - 1) * 100
                            risk_data = {
                                'risk_score': min(100, max(0, price_volatility * 2)),
                                'volatility': min(100, price_volatility),
                                'performance': min(100, max(0, price_trend + 50)),
                                'sentiment': 50 + (price_trend * 0.5),
                                'liquidity': min(100, max(20, 100 - volume_volatility))
                            }
                            visualizations_dict['risk_gauge'] = risk_gauge.create_risk_gauge(risk_data['risk_score'], "Overall Risk")
                            visualizations_dict['volatility_gauge'] = risk_gauge.create_volatility_gauge(risk_data['volatility'])
                            visualizations_dict['performance_gauge'] = risk_gauge.create_performance_gauge(risk_data['performance'])
                            visualizations_dict['advanced_dashboard'] = risk_gauge.create_advanced_dashboard(risk_data)
                        except Exception as e:
                            st.warning(f"Error generating gauge visualizations for report: {str(e)}")
                    if len(clean_data_for_components) > 50:
                        try:
                            visualizations_dict['moving_averages'] = tech_indicators.create_moving_averages_chart()
                            visualizations_dict['rsi_chart'] = tech_indicators.create_rsi_chart()
                            visualizations_dict['macd_chart'] = tech_indicators.create_macd_chart()
                            visualizations_dict['bollinger_bands'] = tech_indicators.create_bollinger_bands_chart()
                        except Exception as e:
                            st.warning(f"Error generating technical indicator charts for report: {str(e)}")
                    # Initialize predictions
                    predictions = PricePredictions(clean_data_for_components)
                    # Initialize comparative analysis if available
                    advanced_analytics = None
                    if st.session_state.current_data is not None and st.session_state.previous_data is not None:
                        try:
                            advanced_analytics = EnhancedComparativeAnalysis(
                                st.session_state.current_data, 
                                st.session_state.previous_data
                            )
                        except Exception as e:
                            st.warning(f"Error initializing comparative analysis for report: {str(e)}")
                    # Generate report
                    report_generator = HTMLReportGenerator()
                    with st.spinner("Generating comprehensive HTML report..."):
                        report_content = report_generator.generate_comprehensive_report(
                            stock_symbol=stock_symbol,
                            historical_data=clean_data_for_components,
                            tech_indicators=tech_indicators,
                            analytics=analytics,
                            visualizations=viz,
                            predictions=predictions,
                            advanced_analytics=advanced_analytics,
                            additional_figures=visualizations_dict,
                            report_type="full"
                        )
                    # Provide download button
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_content,
                        file_name=f"Financial_Analysis_Report_{stock_symbol}_{dt.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    st.success("✅ Report generated successfully! Download the HTML file to view interactive charts.")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Please ensure all required data is available and try again.")

# Main app execution
def main():
    st.title("📊 Premium Financial Analysis Dashboard")
    st.markdown("A comprehensive platform for stock market analysis with advanced visualizations and predictions")
    tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📊 Phase 1: Comparative Analysis", "🔮 Phase 2: Deep Analysis & Advanced Analytics"])
    with tab1:
        data_upload_section()
    with tab2:
        phase1_comparative_analysis_section()
    with tab3:
        st.subheader("🔍 Phase 2: Deep Stock Analysis")
        phase2_deep_analysis_section()
        st.markdown("---")
        st.subheader("🔮 Advanced Analytics & Predictions")
        advanced_analytics_section()

if __name__ == "__main__":
    main()
