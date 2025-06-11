import streamlit as st
import pandas as pd
import numpy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Page configuration - MUST be first
st.set_page_config(
    page_title="Premium Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = None
if 'comparative_analysis' not in st.session_state:
    st.session_state.comparative_analysis = None
if 'yfinance_data' not in st.session_state:
    st.session_state.yfinance_data = None

def data_upload_section():
    """Enhanced Data Upload & Processing section."""
    st.header("📁 Data Upload & Processing")
    st.markdown("Upload your stock data files for comprehensive Phase 1 & Phase 2 analysis")
    
    processor = DataProcessor()
    
    # Current Data Upload
    st.subheader("📊 Current Stock Data")
    current_file = st.file_uploader(
        "Upload Current Stock Data (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="current_data_file",
        help="Upload your current stock trading data with columns like Symbol, Name, Last Sale, % Change, etc."
    )
    
    if current_file is not None:
        try:
            with st.spinner("Processing current stock data..."):
                current_data, quality_report = processor.process_daily_data(current_file)
                
                if current_data is not None:
                    st.session_state.current_data = current_data
                    st.session_state.data_quality_report = quality_report
                    
                    st.success(f"✅ Current data loaded successfully! ({len(current_data)} stocks)")
                    
                    # Data quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Stocks", len(current_data))
                    with col2:
                        # Fix valid symbols count
                        if 'Symbol' in current_data.columns:
                            valid_count = len(current_data[current_data['Symbol'].notna() & 
                                                          (current_data['Symbol'].astype(str).str.strip() != '') & 
                                                          (current_data['Symbol'].astype(str).str.strip() != 'nan')])
                        else:
                            valid_count = 0
                        st.metric("Valid Symbols", valid_count)
                    with col3:
                        st.metric("Data Completeness", f"{quality_report.get('completeness_score', 0):.1f}%")
                    with col4:
                        st.metric("Quality Score", f"{quality_report.get('overall_quality', 0):.1f}/10")
                    
                    # Show sample data
                    with st.expander("📋 Sample Data Preview"):
                        st.dataframe(current_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process current data file. Please check the format.")
                    
        except Exception as e:
            st.error(f"Error processing current data: {str(e)}")
    
    # Previous Data Upload
    st.subheader("📈 Historical Stock Data")
    previous_file = st.file_uploader(
        "Upload Previous Stock Data (Excel/CSV)",  # Fixed string and removed redundant text
        type=["xlsx", "xls", "csv"],
        key="previous_data_file",
        help="Upload previous period stock data for Phase 1 comparative analysis"
    )
    
    if previous_file is not None:
        try:
            with st.spinner("Processing previous stock data..."):
                previous_data, _ = processor.process_daily_data(previous_file)
                
                if previous_data is not None:
                    st.session_state.previous_data = previous_data
                    st.success(f"✅ Previous data loaded successfully! ({len(previous_data)} stocks)")
                    
                    # Show sample data
                    with st.expander("📋 Previous Data Preview"):
                        st.dataframe(previous_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process previous data file. Please check the format.")
                    
        except Exception as e:
            st.error(f"Error processing previous data: {str(e)}")
    
    # Process Button - Show when both files are uploaded
    if st.session_state.current_data is not None and st.session_state.previous_data is not None:
        st.success("🎉 Both datasets ready for analysis")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            if st.button("🚀 Start Analysis"):
                st.session_state.comparative_analysis = ComparativeAnalysis(
                    st.session_state.current_data, st.session_state.previous_data
                )
                st.success("✅ Analysis completed! Navigate to Phase 1: Comparative Analysis to see results.")
                st.balloons()
    
    # Historical Data Upload
    st.subheader("📉 Historical Price Data (Optional)")
    historical_file = st.file_uploader(
        "Upload Historical Price Data (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
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
                    
                    if extracted_symbol:
                        st.info(f"📊 Detected symbol: {extracted_symbol}")
                    
                    # Show sample data
                    with st.expander("📋 Historical Data Preview"):
                        st.dataframe(historical_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process historical data file. Please check the format.")
                    
        except Exception as e:
            st.error(f"Error processing historical data: {str(e)}")
    
    # Data Status Dashboard
    if any([st.session_state.current_data, st.session_state.previous_data, st.session_state.historical_data]):
        st.markdown("---")
        st.subheader("📊 Data Status Dashboard")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            if st.session_state.current_data is not None:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Current Data</h4>
                    <p>Ready for analysis</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>⏳ Current Data</h4>
                    <p>Not uploaded</p>
                </div>
                """, unsafe_allow_html=True)
        
        with status_col2:
            if st.session_state.previous_data is not None:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Previous Data</h4>
                    <p>Ready for Phase 1</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>⏳ Previous Data</h4>
                    <p>Upload for Phase 1</p>
                </div>
                """, unsafe_allow_html=True)
        
        with status_col3:
            if st.session_state.historical_data is not None:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Historical Data</h4>
                    <p>Ready for analysis</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h4>📈 Historical Data</h4>
                    <p>Optional for advanced analysis</p>
                </div>
                """, unsafe_allow_html=True)

def phase1_comparative_analysis_section():
    """Phase 1: Comprehensive comparative analysis between current and previous stock data."""
    st.header("📊 Phase 1: Comparative Analysis")
    st.markdown("Compare current vs previous stock data for comprehensive market analysis")
    
    if st.session_state.current_data is None or st.session_state.previous_data is None:
        st.warning("⚠️ Please upload both current and previous stock data files in the Data Upload tab first.")
        return
    
    try:
        # Initialize comparative analysis
        with st.spinner("Performing comparative analysis..."):
            comp_analysis = ComparativeAnalysis(st.session_state.current_data, st.session_state.previous_data)
            merged_data = comp_analysis.merged_data if hasattr(comp_analysis, 'merged_data') else pd.DataFrame()
            
            if merged_data is None or merged_data.empty:
                st.error("No matching stocks found between current and previous data. Please check Symbol columns.")
                return
        
        # Performance Summary
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
        
        # Top/Bottom Performers
        performers_col1, performers_col2 = st.columns(2)
        
        with performers_col1:
            st.subheader("🏆 Top 5 Performers")
            if merged_data is not None and not merged_data.empty and 'Price_Change_Pct' in merged_data.columns:
                valid_data = merged_data.dropna(subset=['Price_Change_Pct'])
                if not valid_data.empty:
                    top_performers = valid_data.nlargest(5, 'Price_Change_Pct')
                    for idx, row in top_performers.iterrows():
                        symbol = row.get('Symbol', 'N/A')
                        change_pct = row.get('Price_Change_Pct', 0)
                        st.success(f"🟢 **{symbol}**: {change_pct:.2f}%")
                else:
                    st.info("No top performers data available")
            else:
                st.info("No top performers data available")
        
        with performers_col2:
            st.subheader("📉 Bottom 5 Performers")
            if merged_data is not None and not merged_data.empty and 'Price_Change_Pct' in merged_data.columns:
                valid_data = merged_data.dropna(subset=['Price_Change_Pct'])
                if not valid_data.empty:
                    bottom_performers = valid_data.nsmallest(5, 'Price_Change_Pct')
                    for idx, row in bottom_performers.iterrows():
                        symbol = row.get('Symbol', 'N/A')
                        change_pct = row.get('Price_Change_Pct', 0)
                        st.error(f"🔴 **{symbol}**: {change_pct:.2f}%")
                else:
                    st.info("No bottom performers data available")
            else:
                st.info("No bottom performers data available")
        
        # Performance Dashboard
        st.subheader("📊 Performance Dashboard")
        try:
            dashboard_fig = comp_analysis.create_performance_dashboard()
            if dashboard_fig and hasattr(dashboard_fig, 'data') and dashboard_fig.data:
                st.plotly_chart(dashboard_fig, use_container_width=True)
        except:
            st.info("Advanced dashboard temporarily unavailable. Analysis continues below.")
        
        # Professional Risk Assessment Dashboard
        st.subheader("🎯 Professional Risk Assessment")
        
        with st.expander("📚 How to Read Risk Assessment Gauges", expanded=False):
            st.markdown("""
            **Professional Risk Gauges:** These visualizations provide instant insights into your portfolio's risk profile.
            
            **Three Key Metrics:**
            - **Market Risk (Left):** Overall risk level based on price volatility patterns
            - **Volatility Index (Center):** Measures price fluctuation intensity over time  
            - **Performance Score (Right):** Composite performance rating across multiple factors
            
            **Reading the Gauges:**
            - **Green Zone (0-30%):** Conservative, stable investments
            - **Yellow Zone (30-60%):** Moderate risk, balanced approach
            - **Orange Zone (60-80%):** Higher risk, growth-focused
            - **Red Zone (80-100%):** High risk, aggressive investments
            
            **Real-time Example:** If Market Risk shows 45%, Volatility shows 35%, and Performance shows 72%, your portfolio has moderate risk with good returns.
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
                    perf_fig = risk_gauge.create_performance_gauge(risk_data['performance'])
                    st.plotly_chart(perf_fig, use_container_width=True)
        except Exception as e:
            st.info("Risk assessment calculations temporarily unavailable.")
        
        # Enhanced Stock Analysis with Filtering
        st.subheader("📊 Detailed Stock Analysis")
        
        all_stocks_df = merged_data.copy()
        
        if not all_stocks_df.empty and 'Price_Change_Pct' in all_stocks_df.columns:
            valid_data = all_stocks_df[all_stocks_df['Price_Change_Pct'].notna()].copy()
            
            if not valid_data.empty:
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    performance_filter = st.selectbox(
                        "Filter by Performance",
                        ["All Stocks", "Gainers Only", "Losers Only", "Top 10 Performers", "Bottom 10 Performers"],
                        help="Filter stocks based on performance"
                    )
                
                with filter_col2:
                    min_change = st.number_input(
                        "Min Change %",
                        value=float(valid_data['Price_Change_Pct'].min()),
                        help="Minimum price change percentage"
                    )
                
                with filter_col3:
                    max_change = st.number_input(
                        "Max Change %", 
                        value=float(valid_data['Price_Change_Pct'].max()),
                        help="Maximum price change percentage"
                    )
                
                filtered_df = valid_data[
                    (valid_data['Price_Change_Pct'] >= min_change) & 
                    (valid_data['Price_Change_Pct'] <= max_change)
                ].copy()
                
                if performance_filter == "Gainers Only":
                    filtered_df = filtered_df[filtered_df['Price_Change_Pct'] > 0]
                elif performance_filter == "Losers Only":
                    filtered_df = filtered_df[filtered_df['Price_Change_Pct'] < 0]
                elif performance_filter == "Top 10 Performers":
                    filtered_df = filtered_df.nlargest(10, 'Price_Change_Pct')
                elif performance_filter == "Bottom 10 Performers":
                    filtered_df = filtered_df.nsmallest(10, 'Price_Change_Pct')
                
                if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty and 'Price_Change_Pct' in filtered_df.columns:
                    filtered_df = filtered_df.sort_values('Price_Change_Pct', ascending=False)
                
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
                    price_current_col = None
                    price_previous_col = None
                    
                    for col in filtered_df.columns:
                        if 'Last Sale_current' in col or 'Price_current' in col:
                            price_current_col = col
                        elif 'Last Sale_previous' in col or 'Price_previous' in col:
                            price_previous_col = col
                    
                    display_columns = ['Symbol', 'Price_Change_Pct']
                    
                    for col in filtered_df.columns:
                        if 'Name_current' in col or 'Company_current' in col:
                            display_columns.insert(1, col)
                            break
                    
                    if price_current_col:
                        display_columns.append(price_current_col)
                    if price_previous_col:
                        display_columns.append(price_previous_col)
                    
                    for col in filtered_df.columns:
                        if 'Volume_current' in col:
                            display_columns.append(col)
                        elif 'Market Cap_current' in col:
                            display_columns.append(col)
                    
                    available_columns = [col for col in display_columns if col in filtered_df.columns]
                    
                    display_data = filtered_df[available_columns].copy()
                    for col in display_data.columns:
                        if display_data[col].dtype in ['float64', 'int64']:
                            display_data[col] = display_data[col].round(2)
                    
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        height=400
                    )
                    
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
        
        # Sector Analysis
        st.subheader("🏭 Sector Analysis")
        sector_analysis = comp_analysis.get_sector_analysis()
        if not sector_analysis.empty:
            st.dataframe(sector_analysis, use_container_width=True)
        else:
            st.info("Sector analysis not available - missing sector data")
        
        # Industry Analysis
        st.subheader("🏢 Industry Analysis (Top 20)")
        industry_analysis = comp_analysis.get_industry_analysis()
        if not industry_analysis.empty:
            st.dataframe(industry_analysis, use_container_width=True)
        else:
            st.info("Industry analysis not available - missing industry data")
        
        # Outlier Detection
        with st.expander("🎯 Outlier Detection", expanded=False):
            outliers = comp_analysis.detect_outliers()
            
            if outliers:
                outlier_col1, outlier_col2 = st.columns(2)
                
                with outlier_col1:
                    st.write("**🚀 Extreme Gainers**")
                    if outliers.get('extreme_gainers'):
                        for stock in outliers['extreme_gainers']:
                            st.success(f"• {stock['Symbol']}: {stock['Price_Change_Pct']:.2f}%")
                    else:
                        st.info("No extreme gainers detected")
                
                with outlier_col2:
                    st.write("**📉 Extreme Losers**")
                    if outliers.get('extreme_losers'):
                        for stock in outliers['extreme_losers']:
                            st.error(f"• {stock['Symbol']}: {stock['Price_Change_Pct']:.2f}%")
                    else:
                        st.info("No extreme losers detected")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = comp_analysis.calculate_correlations()
        if not correlation_matrix.empty:
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Key Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlation analysis not available")

def phase2_deep_analysis_section():
    """Enhanced Phase 2: Deep analysis with custom date ranges and yfinance integration."""
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
        
        selected_stock = st.selectbox(
            "Select Stock for Deep Analysis",
            options=available_stocks,
            help="Choose a stock from your uploaded data for comprehensive analysis"
        )
    
    with col2:
        st.subheader("📅 Time Period Selection")
        
        date_option = st.radio(
            "Choose date range option:",
            ["Predefined Periods", "Custom Date Range"]
        )
        
        if date_option == "Predefined Periods":
            analysis_period = st.selectbox(
                "Analysis Period",
                options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3,
                help="Predefined time periods for analysis"
            )
            start_date, end_date = None, None
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    help="Select the start date for analysis"
                )
            with col_end:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    help="Select the end date for analysis"
                )
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
                    
                    start_date = pd.to_datetime(hist_data.index[0]).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(hist_data.index[-1]).strftime('%Y-%m-%d')
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>📋 {company_name} ({selected_stock})</h3>
                        <p>Data Period: {start_date} to {end_date} ({len(hist_data)} trading days)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        current_price = hist_data['Close'].iloc[-1]
                        latest_date = pd.to_datetime(hist_data.index[-1]).strftime('%m-%d-%y')
                        st.metric("Current Price", f"${current_price:.2f}", help=f"As of {latest_date}")
                    
                    with col2:
                        if len(hist_data) > 1:
                            price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
                            price_change_pct = (price_change / hist_data['Close'].iloc[-2]) * 100
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
                    
                    hist_data_clean = hist_data.copy().reset_index()
                    
                    if 'Date' in hist_data_clean.columns:
                        hist_data_clean['Date'] = pd.to_datetime(hist_data_clean['Date']).dt.strftime('%Y-%m-%d')
                    else:
                        hist_data_clean['Date'] = hist_data_clean.index.strftime('%Y-%m-%d')
                    
                    if 'Adj Close' not in hist_data_clean.columns:
                        hist_data_clean['Adj Close'] = hist_data_clean['Close']
                        st.info("ℹ️ Using Close price for analysis (Adj Close not available)")
                    
                    for col in ['Dividends', 'Stock Splits']:
                        if col not in hist_data_clean.columns:
                            hist_data_clean[col] = 0
                    
                    if 'Date' in hist_data_clean.columns:
                        hist_data_clean = hist_data_clean.rename(columns={'Date': 'Datetime'})
                    hist_data_clean['Date'] = hist_data_clean['Datetime']
                    
                    st.subheader("📊 Advanced Price Visualizations")
                    
                    if len(hist_data_clean) > 0:
                        viz = Visualizations(historical_data=hist_data_clean)
                        
                        candlestick_fig = viz.create_candlestick_chart()
                        if candlestick_fig:
                            st.plotly_chart(candlestick_fig, use_container_width=True, key="phase2_candlestick")
                        
                        price_trends_fig = viz.create_price_trends_chart()
                        if price_trends_fig:
                            st.plotly_chart(price_trends_fig, use_container_width=True, key="phase2_price_trends")
                        
                        volume_fig = viz.create_volume_chart()
                        if volume_fig:
                            st.plotly_chart(volume_fig, use_container_width=True, key="phase2_volume")
                    
                    if len(hist_data_clean) > 50:
                        st.subheader("⚙️ Technical Indicators Dashboard")
                        
                        tech_indicators = TechnicalIndicators(hist_data_clean)
                        
                        ma_chart = tech_indicators.create_moving_averages_chart()
                        if ma_chart:
                            st.plotly_chart(ma_chart, use_container_width=True, key="phase2_ma_chart")
                        
                        tech_col1, tech_col2 = st.columns(2)
                        
                        with tech_col1:
                            rsi_chart = tech_indicators.create_rsi_chart()
                            if rsi_chart:
                                st.plotly_chart(rsi_chart, use_container_width=True, key="phase2_rsi_chart")
                        
                        with tech_col2:
                            macd_chart = tech_indicators.create_macd_chart()
                            if macd_chart:
                                st.plotly_chart(macd_chart, use_container_width=True, key="phase2_macd_chart")
                        
                        bb_chart = tech_indicators.create_bollinger_bands_chart()
                        if bb_chart:
                            st.plotly_chart(bb_chart, use_container_width=True, key="phase2_bb_chart")
                        
                        st.subheader("🎯 Trading Signals Dashboard")
                        signals = tech_indicators.get_trading_signals()
                        
                        signal_cols = st.columns(min(len(signals), 4))
                        for i, (indicator, signal_data) in enumerate(signals.items()):
                            with signal_cols[i % len(signal_cols)]:
                                signal_value = signal_data.get('signal', 'Unknown')
                                signal_strength = signal_data.get('strength', 'Unknown')
                                
                                if 'buy' in signal_value.lower():
                                    st.markdown(f"""
                                    <div class="success-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {signal_strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif 'sell' in signal_value.lower():
                                    st.markdown(f"""
                                    <div class="warning-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {signal_strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>{indicator}</h4>
                                        <p><strong>{signal_value}</strong></p>
                                        <p>Strength: {signal_strength}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    st.subheader("📈 Performance Metrics")
                    
                    if len(hist_data_clean) > 20:
                        returns = hist_data_clean['Close'].pct_change().dropna()
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            total_return = ((hist_data_clean['Close'].iloc[-1] / hist_data_clean['Close'].iloc[0]) - 1) * 100
                            st.metric("Total Return", f"{total_return:.2f}%")
                        
                        with col2:
                            volatility = returns.std() * np.sqrt(252) * 100
                            st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                        
                        with col3:
                            max_drawdown = ((hist_data_clean['Close'].max() - hist_data_clean['Close'].min()) / hist_data_clean['Close'].max()) * 100
                            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                        
                        with col4:
                            if returns.std() != 0:
                                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                        
                        with col5:
                            avg_volume = hist_data_clean['Volume'].mean()
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
                                    st.metric("Sector Avg Change", f"{sector_avg:.2f}%")
                            
                            with comp_col3:
                                if 'Industry_current' in stock_row:
                                    industry_avg = comp_data[comp_data['Industry_current'] == stock_row['Industry_current']]['Price_Change_Pct'].mean()
                                    st.metric("Industry Avg Change", f"{industry_avg:.2f}%")
                        else:
                            st.info("Stock not found in Phase 1 comparative analysis data")
                
            except Exception as e:
                st.error(f"Error fetching data for {selected_stock}: {str(e)}")
                st.info("Please check if the stock symbol is valid and try again.")

def advanced_data_analysis_section():
    """Advanced Analytics: Predictions and Visualizations."""
    st.header("🔮 Advanced Analytics & Predictions")
    st.markdown("Advanced analytics with predictive modeling and visualizations")
    
    if st.session_state.historical_data is None and st.session_state.yfinance_data is None:
        st.warning("⚠️ Please upload or fetch historical data in Phase 2.")
        return
    
    data_source = st.session_state.yfinance_data if st.session_state.yfinance_data is not None else st.session_state.historical_data
    
    if data_source is None or data_source.empty:
        st.error("No valid historical data available")
        return
    
    data_clean = data_source.copy()
    if hasattr(data_clean, 'reset_index'):
        data_clean = data_clean.reset_index()
    
    if 'Adj Close' not in data_clean.columns:
        data_clean['Adj Close'] = data_clean['Close']
    
    for col in ['Dividends', 'Stock Splits']:
        if col not in data_clean.columns:
            data_clean[col] = 0
    
    if 'Date' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Date': 'Datetime'})
        data_clean['Date'] = pd.to_datetime(data_clean['Datetime']).dt.strftime('%Y-%m-%d')
    
    prediction_tab, visualization_tab, insights_tab = st.tabs(["🔮 Price Predictions", "📊 Advanced Visualizations", "💡 Trading Insights"])
    
    with prediction_tab:
        st.subheader("🔮 Price Predictions")
        
        if len(data_clean) > 50:
            predictions = PricePredictions(data_clean)
            
            col1, col2 = st.columns(2)
            
            with col1:
                prediction_days = st.slider("Prediction Days", min_value=1, max_value=30, value=7)
            
            with col2:
                prediction_method = st.selectbox(
                    "Prediction Method",
                    ["technical_analysis", "linear_trend", "moving_average"],
                    format_func=lambda x: {
                        "technical_analysis": "Technical Analysis",
                        "linear_trend": "Linear Trend",
                        "moving_average": "Moving Average"
                    }[x]
                )
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating price predictions..."):
                    predicted_prices = predictions.predict_prices(prediction_days, method=prediction_method)
                    
                    if predicted_prices is not None:
                        pred_chart = predictions.create_prediction_chart(predicted_prices, prediction_days)
                        st.plotly_chart(pred_chart, use_container_width=True, key="predictions_chart")
                        
                        st.subheader("📈 Predicted Prices")
                        
                        pred_dates = [datetime.now() + timedelta(days=i+1) for i in range(prediction_days)]
                        pred_df = pd.DataFrame({
                            'Date': [d.strftime('%Y-%m-%d') for d in pred_dates],
                            'Day': [f"Day {i+1}" for i in range(prediction_days)],
                            'Predicted Price': [f"${price:.2f}" for price in predicted_prices]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            current_price = data_clean['Close'].iloc[-1]
                            predicted_final = predicted_prices[-1]
                            change_pct = ((predicted_final - current_price) / current_price) * 100
                            st.metric("Predicted Change", f"{change_pct:.2f}%")
                        
                        with col2:
                            st.metric("Target Price", f"${predicted_final:.2f}")
                        
                        with col3:
                            confidence = predictions.calculate_confidence()
                            st.metric("Confidence Score", f"{confidence.get('score', 0):.1f}/10")
                        
                        with col4:
                            volatility = confidence.get('volatility', 0)
                            st.metric("Prediction Volatility", f"{volatility:.2f}%")
                        
                        st.info("⚠️ Predictions are for informational purposes only and not financial advice.")
                    else:
                        st.error("Unable to generate predictions. Try a different method.")
        else:
            st.warning("Insufficient data for predictions (need >50 data points)")
    
    with visualization_tab:
        st.subheader("📊 Advanced Visualizations")
        
        with st.expander("📚 Understanding 3D Factor Analysis", expanded=False):
            st.markdown("""
            **3D Factor Analysis** visualizes three financial dimensions:
            
            - **X-Axis (Risk Level):** Volatility measure
            - **Y-Axis (Return Potential):** Expected gains
            - **Z-Axis (Market Correlation):** Stock's market trend alignment
            - **Color Gradient:** Performance score
            
            **Interpretation:**
            - **High Peaks:** Optimal investments
            - **Valleys:** Higher risk areas
            - **Clusters:** Similar characteristics
            """)
        
        if len(data_clean) > 0:
            viz = Visualizations(historical_data=data_clean, daily_data=st.session_state.current_data)
            
            visualization_option = st.selectbox(
                "Select Visualization",
                ["Candlestick Chart", "Price Trends", "Volume Analysis", "Market Overview Dashboard", "3D Factor Analysis"]
            )
            
            if visualization_option == "Candlestick Chart":
                fig = viz.create_candlestick_chart()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif visualization_option == "Price Trends":
                fig = viz.create_price_trends_chart()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif visualization_option == "Volume Analysis":
                fig = viz.create_volume_chart()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif visualization_option == "3D Factor Analysis":
                if len(data_clean) >= 20:
                    returns = data_clean['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100
                    total_return = ((data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0]) - 1) * 100
                    
                    risk_levels = np.linspace(0, 100, 20)
                    return_levels = np.linspace(-50, 150, 20)
                    X, Y = np.meshgrid(risk_levels, return_levels)
                    Z = np.exp(-((X - volatility)**2 + (Y - total_return)**2) / 1000)
                    
                    fig = go.Figure(data=[go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale='Viridis',
                        colorbar=dict(title="Performance Score")
                    )])
                    
                    fig.update_layout(
                        title='3D Factor Analysis: Risk vs Return vs Performance',
                        scene=dict(
                            xaxis_title='Risk Level (%)',
                            yaxis_title='Return Potential (%)',
                            zaxis_title='Performance Score'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Risk", f"{volatility:.1f}%")
                    with col2:
                        st.metric("Total Return", f"{total_return:.1f}%")
                    with col3:
                        performance_score = min(100, max(0, 50 + total_return/2))
                        st.metric("Performance Score", f"{performance_score:.1f}")
                else:
                    st.warning("Need more data points for 3D analysis (minimum 20 days)")
            elif visualization_option == "Market Overview Dashboard":
                if st.session_state.current_data is not None:
                    fig = viz.create_market_overview_dashboard()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    with insights_tab:
        st.subheader("💡 Trading Insights")
        
        if len(data_clean) >= 50:
            tech_indicators = TechnicalIndicators(data_clean)
            signals = tech_indicators.get_trading_signals()
            analytics = Analytics(historical_data=data_clean)
            
            st.markdown("### Trading Signals")
            
            signal_counts = {"buy": 0, "sell": 0, "hold": 0}
            for signal, data in signals.items():
                signal_val = data.get('signal', '').lower()
                if 'buy' in signal_val:
                    signal_counts["buy"] += 1
                elif 'sell' in signal_val:
                    signal_counts["sell"] += 1
                else:
                    signal_counts["hold"] += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Buy Signals", signal_counts["buy"])
            with col2:
                st.metric("🔴 Sell Signals", signal_counts["sell"])
            with col3:
                st.metric("🟡 Neutral Signals", signal_counts["hold"])
            
            st.markdown("### Signal Analysis")
            for signal, data in signals.items():
                signal_value = data.get('signal', '')
                strength = data.get('strength', '')
                
                with st.expander(f"{signal} - {signal_value}"):
                    st.markdown(f"**Current Signal:** {signal_value}")
                    st.markdown(f"**Signal Strength:** {strength}")
                    
                    if signal == 'RSI':
                        rsi_val = tech_indicators.calculate_rsi()[-1]
                        st.markdown(f"**Current RSI:** {rsi_val:.2f}")
                        if rsi_val > 70:
                            st.warning("Stock may be overbought")
                        elif rsi_val < 30:
                            st.success("Stock may be oversold")
            
            st.markdown("### Strategy Recommendations")
            strategies = analytics.get_trading_strategies(signals)
            
            if strategies:
                for i, strategy in enumerate(strategies):
                    with st.expander(f"Strategy {i+1}: {strategy.get('name', 'N/A')}"):
                        st.markdown(f"**Type:** {strategy.get('type', 'N/A')}")
                        st.markdown(f"**Risk:** {strategy.get('risk_level', 'N/A')}")
                        st.markdown(f"**Timeframe:** {strategy.get('time_horizon', 'N/A')}")
                        st.markdown(f"**Description:** {strategy.get('description', 'N/A')}")
            
            st.markdown("### Risk Assessment")
            risk_metrics = analytics.get_risk_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Beta", f"{risk_metrics.get('beta', 'N/A'):.2f}")
            with col2:
                st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 'N/A'):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 'N/A'):.2f}%")
            with col4:
                st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 'N/A'):.2f}%")
        else:
            st.markdown("Need at least 50 days of data for reliable insights.")

# Custom CSS
st.markdown("""
<style>
.main > div {
    padding: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.success-card {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.warning-card {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Financial Analysis Dashboard")
st.markdown("### Comprehensive Stock Analysis with Technical Indicators")

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Data Upload", 
    "📊 Phase 1: Comparative Analysis", 
    "📈 Phase 2: Deep Analysis",
    "🔮 Advanced Analytics"
])

with tab1:
    data_upload_section()

with tab2:
    phase1_comparative_analysis()

with tab3:
    phase2_deep_analysis()

with tab4:
    advanced_data_analysis()

if __name__ == "__main__':
    main()
