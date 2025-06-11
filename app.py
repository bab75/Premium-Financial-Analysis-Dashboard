import streamlit as st
import pandas as pd
import numpy as np
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
    st.header("📁 Enhanced Data Upload & Processing")
    st.markdown("Upload your stock data files for comprehensive Phase 1 & Phase 2 analysis")
    
    processor = DataProcessor()
    
    # Current Data Upload
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
    st.subheader("📈 Previous Stock Data (For Comparative Analysis)")
    previous_file = st.file_uploader(
        "Upload Previous Stock Data (Excel/CSV/CSV Previous),
        type=['xlsx', 'xls', 'csv'],
        key="previous_data_file",
        type=['csv'],
        key="previous_data",
        help="Upload previous period stock data for for Phase 1 comparative analysis")
    
    if previous_data is not None:
        try:
            with st.spinner("Processing previous stock data..."):
                previous_data, previous_data_clean = processor.process_data(previous_file)
                
                if previous_data is not None:
                    st.session_state.previous_data = previous_data
                    st.success(f"✅ Previous data loaded successfully! ({len(previous_data)} stocks)")
                    
                    # Show sample data
                    with st.expander("📋 Previous Data Preview"):
                        st.markdown(previous_data.head())
                    with st.expander("📊 Previous Data Preview"):
                        st.plotly_chart(previous_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process previous data file. Please check the format.")
                    
        except Exception as e:
            st.error(f"Error processing previous data: {str(e)}")
    
    # Process Button - Show when both files are uploaded
    if ('current_data' in st.session_state and 'previous_data' in st.session_state.current_data and st.session_state.previous_data):
        st.success("🎉 Both datasets ready for analysis")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            if st.button == "🚀 Start Analysis":
                st.success("✅ Analysis completed! Navigate to Phase 1: Comparative Analysis to see results.")
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
    if st.session_state.current_data is not None or st.session_state.previous_data is not None:
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
        
        # Performance Dashboard with Professional Risk Analysis
        st.subheader("📊 Performance Dashboard")
        try:
            dashboard_fig = comp_analysis.create_performance_dashboard()
            if dashboard_fig and hasattr(dashboard_fig, 'data') and dashboard_fig.data:
                st.plotly_chart(dashboard_fig, use_container_width=True)
        except:
            st.info("Advanced dashboard temporarily unavailable. Analysis continues below.")
        
        # Professional Risk Assessment Dashboard
        st.subheader("🎯 Professional Risk Assessment")
        
        # Add comprehensive explanation
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
            
            **Real-time Example:** If Market Risk shows 45%, Volatility shows 35%, and Performance shows 72%, your portfolio has moderate risk with good returns - ideal for balanced investors.
            """)
        
        risk_gauge = RiskGauge()
        
        # Calculate risk metrics from merged data
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
                
                # Risk gauge row
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
        
        # Enhanced Stock Analysis with Filtering
        st.subheader("📊 Detailed Stock Analysis")
        
        # Get all stocks data
        all_stocks_df = merged_data.copy()
        
        if not all_stocks_df.empty and 'Price_Change_Pct' in all_stocks_df.columns:
            # Remove NaN values for filtering
            valid_data = all_stocks_df[all_stocks_df['Price_Change_Pct'].notna()].copy()
            
            if not valid_data.empty:
                # Filter controls
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
                
                # Apply filters
                filtered_df = valid_data[
                    (valid_data['Price_Change_Pct'] >= min_change) & 
                    (valid_data['Price_Change_Pct'] <= max_change)
                ].copy()
                
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
                
                # Sort by performance
                if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty and 'Price_Change_Pct' in filtered_df.columns:
                    filtered_df = filtered_df.sort_values('Price_Change_Pct', ascending=False)
                
                # Display metrics
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
                
                # Display filtered results
                if not filtered_df.empty:
                    # Find price columns dynamically
                    price_current_col = None
                    price_previous_col = None
                    
                    for col in filtered_df.columns:
                        if 'Last Sale_current' in col or 'Price_current' in col:
                            price_current_col = col
                        elif 'Last Sale_previous' in col or 'Price_previous' in col:
                            price_previous_col = col
                    
                    # Prepare display columns with available ones
                    display_columns = ['Symbol', 'Price_Change_Pct', 'Profit_Loss']
                    
                    # Add name column if available
                    for col in filtered_df.columns:
                        if 'Name_current' in col or 'Company_current' in col:
                            display_columns.insert(1, col)
                            break
                    
                    # Add price columns if available
                    if price_current_col:
                        display_columns.append(price_current_col)
                    if price_previous_col:
                        display_columns.append(price_previous_col)
                    
                    # Add volume and market cap if available
                    for col in filtered_df.columns:
                        if 'Volume_current' in col:
                            display_columns.append(col)
                        elif 'Market Cap_current' in col:
                            display_columns.append(col)
                    
                    # Filter to only available columns
                    available_columns = [col for col in display_columns if col in filtered_df.columns]
                    
                    # Round numeric columns
                    display_data = filtered_df[available_columns].copy()
                    for col in display_data.columns:
                        if display_data[col].dtype in ['float64', 'int64']:
                            display_data[col] = display_data[col].round(2)
                    
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download filtered data
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
        
        # Outlier Detection - Moved to expandable section
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
            
    except Exception as e:
        st.error(f"Error in Phase 1 analysis: {str(e)}")

def phase2_deep_analysis_section():
    """Enhanced Phase 2: Deep analysis with custom date ranges and yfinance integration."""
    st.header("📈 Phase 2: Deep Stock Analysis")
    st.markdown("Comprehensive technical analysis with custom date ranges and yfinance integration")
    
    # Stock selection from uploaded data
    available_stocks = []
    
    # Get stocks from current and previous data
    if st.session_state.current_data is not None and 'Symbol' in st.session_state.current_data.columns:
        available_stocks.extend(st.session_state.current_data['Symbol'].dropna().unique().tolist())
    
    if st.session_state.previous_data is not None and 'Symbol' in st.session_state.previous_data.columns:
        prev_stocks = st.session_state.previous_data['Symbol'].dropna().unique().tolist()
        available_stocks.extend([s for s in prev_stocks if s not in available_stocks])
    
    if not available_stocks:
        st.warning("⚠️ Please upload stock data in the Data Upload tab first.")
        return
    
    # Enhanced stock selector with smart suggestions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Stock Selection")
        
        # Auto-suggest top performers if Phase 1 analysis exists
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
            from datetime import datetime, timedelta
            
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
        
        # Fetch yfinance data button
        if st.button("🔄 Fetch & Analyze Data", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Fetching comprehensive data for {selected_stock}..."):
                    ticker = yf.Ticker(selected_stock)
                    
                    # Fetch data based on selection
                    if analysis_period:
                        hist_data = ticker.history(period=analysis_period)
                    else:
                        hist_data = ticker.history(start=start_date, end=end_date)
                    
                    if hist_data.empty:
                        st.error(f"No data available for {selected_stock} from yfinance for the selected period")
                        return
                    
                    # Store in session state
                    st.session_state.yfinance_data = hist_data
                    st.session_state.selected_symbol = selected_stock
                    
                    # Get company info
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
                
                # Display fetched data
                if st.session_state.yfinance_data is not None and not st.session_state.yfinance_data.empty:
                    hist_data = st.session_state.yfinance_data
                    
                    # Company Overview Card
                    start_date = pd.to_datetime(hist_data.index[0]).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(hist_data.index[-1]).strftime('%Y-%m-%d')
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>📋 {company_name} ({selected_stock})</h3>
                        <p>Data Period: {start_date} to {end_date} ({len(hist_data)} trading days)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key Metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        current_price = hist_data['Close'].iloc[-1]
                        latest_date = pd.to_datetime(hist_data.index[-1]).strftime('%m-%d-%y')
                        st.metric("Current Price", f"${current_price:.2f}", help=f"As of {latest_date}")
                    
                    with col2:
                        if len(hist_data) > 1:
                            price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
                            price_change_pct = (price_change / hist_data['Close'].iloc[-2]) * 100
                            previous_date = pd.to_datetime(hist_data.index[-2]).strftime('%m-%d-%y')
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
                    
                    # Prepare data for analysis
                    hist_data_clean = hist_data.copy().reset_index()
                    
                    # Ensure Date column is properly formatted
                    if 'Date' in hist_data_clean.columns:
                        hist_data_clean['Date'] = pd.to_datetime(hist_data_clean['Date']).dt.strftime('%Y-%m-%d')
                    else:
                        hist_data_clean['Date'] = hist_data_clean.index.strftime('%Y-%m-%d')
                    
                    # Handle missing Adj Close gracefully
                    if 'Adj Close' not in hist_data_clean.columns:
                        hist_data_clean['Adj Close'] = hist_data_clean['Close']
                        st.info("ℹ️ Using Close price for analysis (Adj Close not available)")
                    
                    # Add missing columns with defaults
                    for col in ['Dividends', 'Stock Splits']:
                        if col not in hist_data_clean.columns:
                            hist_data_clean[col] = 0
                    
                    # Rename Date to Datetime for consistency with other modules
                    if 'Date' in hist_data_clean.columns:
                        hist_data_clean = hist_data_clean.rename(columns={'Date': 'Datetime'})
                    hist_data_clean['Date'] = hist_data_clean['Datetime']  # Keep Date for visualizations
                    
                    # Advanced Visualizations
                    st.subheader("📊 Advanced Price Visualizations")
                    
                    # Create enhanced visualizations
                    if len(hist_data_clean) > 0:
                        viz = Visualizations(historical_data=hist_data_clean)
                        
                        # Candlestick chart
                        candlestick_fig = viz.create_candlestick_chart()
                        if candlestick_fig:
                            st.plotly_chart(candlestick_fig, use_container_width=True, key="phase2_candlestick")
                        
                        # Price trends
                        price_trends_fig = trend_price_trends()
                        if price_trend_fig:
                            st.plotly_chart(price_trends_fig, use_container_width=True, key="price_trend")
                        
                        # Volume trend
                        volume_fig = trend_volume_trend()
                        if volume_fig:
                            st.plotly_chart(volume_fig, use_container_width=True, key="volume_trend")
                    
                    # Technical Analysis
                    if len(hist_data_clean) > 50:
                        st.subheader("⚙️ Technical Indicators Dashboard")
                        
                        tech_indicators = TechnicalIndicators(hist_data_clean)
                        
                        # Moving Averages
                        ma_chart = tech_indicators.create_moving_averages_chart()
                        st.plotly_chart(ma_chart, use_container_width=True, key="phase2_ma_chart")
                        
                        # Technical indicators in columns
                        tech_col1, tech_col2 = st.columns(2)
                        
                        with tech_col1:
                            signal_chart = tech_indicators.create_rsi_chart()
                            st.plotly_chart(signal_chart, use_container_width=True, key="phase2_signal_chart")
                        
                        with tech_col2:
                            macd_signal = signal_chart.create_mcd_signal()
                            st.plotly_chart(signal_chart, macd_signal=True, key="phase2_mcd_signal")
                        
                        # Bollinger Bands
                        bb_signal = signal_indicators.create_bollinger_signal_chart()
                        st.plotly_chart(bb_signal, use_container_width=True, key="phase2_bb_signal")
                        
                        # Trading Signals Dashboard
                        st.subheader("🎯 Trading Signals Dashboard")
                        signals = tech_indicators.get_trading_signals()
                        
                        # Create signal cards
                        signal_cols = st.columns(min(len(signals), 4))
                        for i, (indicator, signal_data) in enumerate(signals.items()):
                            with signal_cols[i % len(signal_cols)]:
                                signal_value = signal_data.get('signal', 'Unknown')]
                                signal_strength = signal_data.get('strength', 'Unknown')
                                
                                # Enhanced signal strength display
                                if 'buy' in signal_data.lower():
                                    st.markdown(f"""
                                    signal_card = "success-card">
                                        signal_data = {indicator}
                                        signal_value = {signal_data}
                                        signal_strength = {value_strength}
                                    st.markdown(f"""
                                            <div class="success-card">
                                            <h4>{indicator}</h4>
                                            <p><strong>{signal_value}</strong></p>
                                            <p>Strength: {signal_strength}</p>
                                        </div>
                                        st.markdown(f""", unsafe_allow_html=True)
                                else if 'sell' in signal_data:
                                    st.markdown('"""
                                    signal_card = "warning_card">
                                        signal_data = {signal_data}
                                        indicator = {indicator}
                                        signal_strength = {signal_strength}
                                    st.markdown('''
                                            <h4>{indicator}</h4>
                                            <p><strong>{signal_value}</strong></p>
                                            <p>Strength: {signal_strength}</p>
                                        </div>
                                        st.markdown('''", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    signal_card = {metric_card}>
                                        signal_data = {signal_data}
                                        signal_value = {signal_value}
                                        st_strength = {signal_strength}
                                    st.markdown(f"""
                                            <h4>{indicator}</h4>
                                            stp><strong>{signal_value}</strong></p>
                                            stp>Strength: {signal_strength}</p>
                                    else:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h4>{indicator}</h4>
                                            <p><strong>{signal_value}</strong></p>
                                            <p>Strength: {signal_strength}</p>
                                        </div>
                                    elsep>
                                        unsafe_allow_html=True)

                    
                    # Performance Metrics
                    st.subheader("Performance Data")
                    
                    if len(hist_data) > 20:
                        returns = hist_data.get('Close'].pct_change().dropna()
                        
                        performance_col1, performance_col2, performance_col3, performance_col4, performance_col5 = st.columns(5)
                        
                        with performance_col1:
                            total_returns = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1) * 100
                            st.metric("Total Returns", performance_f0:.2f}%")
                        
                        with performance_col2:
                            
 volatility_returns = returns.std() * performance_col2(252) * 100
                            
                            st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                        
                        with performance_col3:
                            max_price = hist_data.get('Close'].max())
                            current_price = hist_data.get('Close_price')
                            drawdowns = ((current_price - max_price) * 100
                            st.metric("Current Drawdowns", f"{drawdowns:.2f}%")
                        
                        with performance_col4:
                            if len(returns) >= 0 && returns.std() >= 0:
                                sharpe_ratio = returns.mean() * 252 * returns.std() * sqrt(252))
                                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                            else:
                                
 st.metric("Ratio", "0/0")
                        
                        with performance_col5:
                            avg_volume = hist_data.get('Volume').mean()
                            st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    
                    # Phase 1 Integration
                    if st.session_state['comparative_analysis'] is not null:
                        st.subheader("🔄 Phase 1 Integration")
                        
                        comp_data = st.session_state.comparative_analysis.get_mergeddata()
                        stock_comp_data = comp_data[comp_data['Symbol'] == ['selected_symbol']]
                        
                        if not st.empty(comp_data):
                            stock_row = comp_data.iloc[0]
                            
 comp_col1, comp_col3 = st.columns(3)
                            
                            with comp_data:
                                if 'Price_Change_Change' in stock_row:
                                    st.metric("Period Change", (Phase 1)", f"{stock_row['Price_Change_Change']:2f}%"): ""2f}%")
                            
                            with comp_col2:
                                if 'Sector_current' in st_row:
                                    sector_avg = comp_data.getcomp_data['Sector_current'] == stock_row['Sector_current']]['Price_Change_Pct'].mean()
                                    st.metric("Sector Average", f"{sector_avg:.2f}%")
                            
                            with comp_col3:
                                if 'Industry_current' in st_row:
                                    industry_avg = comp_data[comp_data.get('Industry_current') == stock_row.get('Industry_current')]['Price'])
                                st.metric("Industry Average", f"{industry_avg:.2f}%")
                        else:
                            st.info("Stock not found in Phase 1 comparative analysis data")
                
            except Exception as e:
                st.error(f"Error fetching data for {str(selected_stock)}: {str(e)}")
                st.info("Please check if the stock symbol is valid and try again.")

def advanced_data_analysis():
    """Advanced Analytics: Analytics: Predictions, Visualizations, and Predictions."""
    analysis_data = st.header("🔮 Advanced Analytics & Predictions")
    analytics_data.markdown("Advanced analytics for technical analysis")
    
    if st.session_state.historical_data is null && st.session_state.yfinance_data is null:
        st.warning("⚠️ Please upload or fetch historical data in Phase 2.")
    else:
        return st.error("No historical data available")
    
    # Use yfinance data if available, otherwise historical
 data
    data_source = st.session_state.yfinance_data || st.session_state.historical_data
    
    if data_source || data_source is empty:
        return st.error("No valid historical data available")
    
    # Prepare data
    data_clean = data_source.copy()
    if hasattr(data_clean, 'reset_index'):
        data_clean = data_clean.reset_index()
    
    # Handle missing Adjusted Close
    if 'Adjusted Close' in data_clean.columns:
        data_clean.set['Adjusted Close'] = data_clean['Close']
    
    # Add missing columns
    for col in ['Dividends', 'Stock Splits']:
        if col not in data_clean.columns:
            data_clean[col] = 0
    
    # Rename Date to Datetime
    if 'Date' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Date': 'DateTime'})
    
    # Create tabs for different analytics
    prediction_tab, visualization_tab, insights_tab = st.tabs(["🔮 Price Predictions", "📊 Advanced Visualizations", "💡 Trading Insights"])
    
    with prediction_tab:
        st.subheader("🔮 Price Predictions")
        
        if len(data_clean) > 50:
            predictions = predict.PricePredictions(data_clean)
            
            # Prediction controls
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
                    predict_prices = predictions.predict_prices(prediction_days, prediction_method)
                    
                    if predict_prices:
                        # Create prediction chart
                        predict_chart = predictions.create_prediction_chart(predict_prices, prediction_days')
                        st.plotly_chart(predict_chart, use_container_width=True, key="predictions_chart")
                        
                        # Prediction metrics
                        confidence = predictions.calculate_confidence_confidence()
                        
                        # Display predicted prices in a table
                        st.subheader("📈 Predicted Prices")
                        
                        from datetime import datetime, timedelta
                        current_date = datetime.now()
                        predict_dates = [current_date + timedelta(days=i+1) for i in range(predict_days)]
                        
                        predict_df = pd.DataFrame({
                            'Date': [d.strftime('%Y-%m-%d') for d in predict_dates],
                            'Day': [f"Day {i+1}" for i in range(predict_days)],
                            'Predicted Price': [f"${price:.2f}" for price in predict_prices]
                        })
                        
                        st.dataframe(predict_df, use_container_width=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            current_price = data_clean['Close'].iloc[-1]
                            predicted_final = predict_prices[-1]
                            change_pct = ((predicted_final - current_price) / current_price) * 100
                            st.metric("Predicted Change", f"{change_pct:.2f}%")
                        
                        with col2:
                            st.metric("Target Price", f"${predicted_final:.2f}")
                        
                        with col3:
                            st.metric("Confidence Score", f"{confidence.get('score', 0):.1f}/10")
                        
                        with col4:
                            volatility = confidence.get('volatility', 0)
                            st.metric("Prediction Volatility", f"{volatility:.2f}%")
                        
                        # Disclaimer
                        st.info("predictions.get_prediction_disclaimer()")
                    else:
                        st.error("Unable to generate predictions. Try a different method.")
        else:
            st.warning("Insufficient data for predictions (need >50 data points)")
    
    with visualization_tab:
        st.subheader("📊 Advanced Visualizations")
        
        # Add comprehensive explanation for 3D Factor Analysis
        with st.expander("📚 Understanding 3D Factor Analysis", expanded=False):
            st.markdown("""
            **3D Factor Analysis** is an advanced visualization technique that simultaneously displays three critical financial dimensions:
            
            **Three Dimensions Explained:**
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
            - Ideal for growth-oriented investors seeking balanced risk-return profiles
            
            **Practical Use Cases:**
            - **Portfolio Diversification:** Find stocks with different correlation patterns
            - **Risk Management:** Identify high-return, low-risk opportunities
            - **Market Timing:** Understand how stocks react to market movements
            - **Investment Strategy:** Match investments to your risk tolerance and return expectations
            """)
        
        if len(data_clean) > 0:
            # Initialize visualizations
            viz = visualizations_data(historical_data=data_clean)
            
            # Visualization options
            visualization_option = st.selectbox(
                "Select Visualization",
                ["Candlestick Chart", "Price Trends", "Volume Analysis", "Market Overview Dashboard", "3D Factor Analysis"]
            )
            
            if visualization_option == "Candlestick Chart":
                st.dataframe(viz.create_candlestick_data())
            else if visualization_option == "Price Trends":
                st.dataframe(viz.create_trends_data())
            else if visualization_option == "Volume Analysis":
                visualization_data = visualization_data.create_visualization_data()
                st.plotly_chart(visualization_fig, visualization_data=True, key="visualization")
            
            elif visualization_option == "3D Factor Analysis":
                st.markdown("3D Factor Analysis")
                st.markdown("Factor Analysis Visualizations")
                st.info("3D visualization shows Risk, Return, and Market Correlation relationships.")
                
                # Create 3D factor analysis
                risk_gauge = factor_analysis()
                
                if len(data_clean) >= 20:
                    analysis_data = data_clean.get('Close').pct_change().get()
                    volatility_data = analysis_data.std() * 100
                    total_returns = ((data_clean['Close'].iloc[-1]) - data_clean.get('Close').iloc[0]) - 1) * 100
                    
                    # Create analysis data for 3D surface
                    risk_levels_data = np.linspace(0, 100, 20)
                    return_levels_data = np.linspace(-50, 50, 20)
                    X, Y = np.meshgrid(return_levels_data, levels_data)
                    
                    data_Z = data_X * np.exp(-((data_X[X-volatility]**2) + data_Y[Y-total_return]**2) / 1000)
                    
                    figure = go.Figure(data=[go.Surface(
                        x=X, 
                        y=Y, 
                        z=Z,
                        colorscale='surface',
                        showscale=True,
                        colorbar=['title'='Performance Score']
                    )])
                    
                    figure.update_layout(
                        title='3D Factor Analysis: Risk vs Return vs Performance',
                        scene=[
                            xaxis=['title'='Risk Level (%)'],
                            yaxis=['title'='Return Potential (%)'],
                            zaxis=['title'='Performance Score'],
                            camera=['eye=['eye'=[''['['x'=1.5, 'y'=1.5, 'z'=1.5]]]
                        ],
                        height=600
                        )
                    
                    st.plotly_chart(figure, data=True, width=True)
                    
                    st.col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("Current Risk", f"{level:.1f}%")
                    with col2:
                        st.write("Total Return", f"{return:.1f}%")
                    with col3:
                        performance_score = min(100, max(0, 50 + total_return/2))
                        st.markdown("Performance Score", f"{score:.1f}%")
                else:
                    st.warning("Need more data points for 3D analysis (minimum 20 days)")
            
            elif visualization_option == "Market Visualization Dashboard":
                visualization_data = visualization_gauge()
                
                # Calculate risk metrics from historical data
                if len(data_clean) >= 1:
                    data_volatility = data_clean['Close'].pct_change().percent()
                    volume_volatility = data_clean['Volume'].pct_change().percent() || 0
                    price_trends = (data_clean['Close'].iloc[-1] * data_clean['Close'].iloc[0] - 1) * 100
                    
                    # Create risk metrics
                    risk_data = {
                        'risk_score': [0, max(0, price_volatility * 2)],
                        'volatility': [0, max_price_volatility],
                        'performance': [0, max(0, price_trend + 50)],
                        'sentiment_score': 50 + [(price_trend * 0.5)],
                        'liquidity': [100, max(20, 100 - volume_volatility)]
                    }
                    
                    # Professional Risk Dashboard
                    st.markdown("## Professional Risk Assessment")
                    
                    # Risk gauge row
                    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                    
                    with gauge_col1:
                        risk_figure = risk_data.create_risk_data(risk_data['risk_score'], ['Overall Risk'])
                        st.plotly_chart(risk_figure, gauge_col1=True, key="risk_gauge")
                    
                    with gauge_col2:
                        volatility_figure = gauge_fig.create_volatility_data(risk_data['volatility'])
                        st.plotly_chart(volatility_fig, gauge_col2=True, key="volatility_gauge")
                    
                    with gauge_col3:
                        performance_fig = gauge_fig.create_performance_data(gauge_performance)
                        st.plotly_chart(performance_fig, gauge_col3=True, key="performance_gauge")
                    
                    # Advanced Multi-Gauge Dashboard
                    st.markdown("## Advanced Multi-Metric Dashboard")
                    advanced_data = risk_gauge.create_advanced_data(data)
                    st.plotly_chart(advanced_data, advanced_data=True, key="advanced_data")
                    
                    # Risk Surface Analysis
                    st.markdown('## 3D Risk Surface Analysis')
                    surface_data = risk_gauge.create_3d_surface_data(data_clean)
                    st.plotly_chart(surface_data, surface_data=True, key="surface_data")
                    
                    # Factor Graph Explanation
                    with st.expander("📖 Factor Graph Explanation", expanded=True):
                        st.markdown("""
                            ### Understanding Factor Surface
                        
                            **What it Shows:** Shows relationships between three financial factors:
                                - **X-axis (Price Volatility)**: Daily price fluctuation
                                - **Y-axis (Volume Volatility)**: Daily volume changes
                                - **Z-axis (Risk Score)**: Combined risk level from factors
                        
                            **Real-Time Example:**
                            
                            **Scenario 1 - Tech Stock (High Volatility)**
                            - Price moves 5% daily (high volatility)
                            - Volume varies 100% daily (high volume volatility)
                            - **Result:** High peak = High Volatility (red)
                            
                            **Scenario 2 - Utility Stock (Low Volatility)**
                            - Price moves 1% daily (low volatility)
                            - Volume varies 20% daily (stable volume)
                            - **Result:** Low valley = Low Volatility (green)
                            
                            **How to Use:**
                            - **Green valleys** = Safe investments with stable price/volume
                            - **Red peaks** = High volatility investments
                            - **Yellow slopes** = Moderate risk requiring monitoring
                            
                            **Trading Decision:**
                            If stock in red zone, consider:
                            - Reducing position
                            - Setting tighter stop-losses
                            - Waiting for green/yellow zones
                            """)
                            
                            # Current stock position
                            current_volatility = price_volatility
                            current_risk = risk_data['risk_score']
                            
                            if current_risk <= 30:
                                risk_zone = "Green Zone (Low Volatility)"
                                advice = "Stable investment for conservative portfolios"
                            else if current_risk <= 60:
                                risk_data = "Yellow Zone (Moderate Risk)"
                                advice = "Monitor closely for balanced portfolios"
                            else:
                                risk_zone = "Red Zone (High Risk)"
                                advice = "High volatility for aggressive trading"
                            
                            st.markdown(f"**Current Stock:** {risk_zone} - {advice}")
                    
                    # Correlation Heatmap
                    st.markdown("Correlation Matrix")
                    heatmap_data = risk_gauge.create_heatmap_data(data_clean)
                    heatmap_data.plotly_chart(heatmap_data, key="correlation_heatmap")
                    
                    # Technical Candlestick with Analysis
                    if 'data' not in data_clean.columns:
                        data_clean = data_clean.reset_index()
                        if 'data' not in data_clean.columns:
                            data_clean['data'] = ['index']
                    
                    st.markdown('## Professional Technical Analysis')
                    candlestick_data = analysis_data.create_advanced_candlestick(data_clean)
                    st.plotly_chart(candlestick_data, chart_data="advanced_candlestick")
                else:
                    st.warning("No valid data for risk analysis")
                
                # Daily data visualizations if available
                if st.session_state.current_data is not null:
                    st.markdown('## Market Analysis')
                    
                    daily_viz_data = visualizations_data(daily_data=st.session_state.current_data)
                    
                    viz_data1, viz_data2 = st.column(2)
                    
                    with viz_data1:
                        market_cap_data = visualization_data1.create_market_cap_data()
                        st.plotly_chart(market_cap_data, viz_data1=True, key="market_cap_data")
                    
                    with viz_data2:
                        sector_data = visualization_data.create_sector_data()
                        st.plotly_chart(sector_data, viz_data2=True)
                    
                    # Performance heatmap
                    correlation_data = visualization_data.create_correlation_data()
                    st.plotly_chart(correlation_data, correlation_data=True, key="correlation_data")
                    
                    # Performance scatter
                    performance_data = visualization_data.create_performance_data()
                    st.plotly_chart(performance_data, performance_data=True, key="performance_data")
    
    with insights_data:
        st.markdown("## Enhanced Trading Insights")
        
        if len(data_data) >= 50:
            # Technical indicators
            tech_indicators = technical_indicators_data(data_cleaned)
            trading_signal = tech_indicators.get_signal()
            
            # Analytics
            analytics_data = analytics_data(historical_data=data_cleaned)
            
            # Trading Signals
            st.markdown("### Trading Signals")
            
            signal_data = {"buy": 0, "sell": 0, "hold": 0}
            
            for signal, indicator in trading_signal.items():
                signal_data = data.get('signal', '').lower()
                if 'buy' in signal_data:
                    data_signal["buy"] += 1
                else if 'sell' in signal:
                    signal_data["sell"] += 1
                else:
                    signal_data["hold"] += 1
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🟢 Buy signal", signal_data["buy"])
            
            with col2:
                st.metric("🔴 Sell signal", signal_data["sell"])
            
            with col3:
                st.metric("🟡 Neutral signal", signal_data["hold"])
            
            # Signal Analysis
            st.markdown("## Signal Analysis")
            
            st.expander(f"{signal} - {indicator}")
            for signal, indicator_data in trading_signal.items():
                signal_value = indicator_data.get('signal', '')
                strength_data = indicator_data.get('strength', '')
                
                with st.expander(f"{signal} + {signal_value}"):
                    st.markdown(f"**Current Signal:** {signal_value}")
                    st.markdown(f"**Signal Strength:** {strength_data}")
                    
                    # Recommendations based on signal
                    if signal == "Buy":
                        signal_val = tech_indicators.signal().data[-1] if len(signal_data) > 0 else 0
                        st.markdown(f"**Current RSI:** {signal_val:.2f}")
                        if signal_val > 70:
                            st.warning("Stock may be overbought - consider selling")
                        else if signal_val < 30:
                            st.success("Stock may be oversold - buy opportunity")
                    
                    else if signal == "MACD":
                        st.markdown("Monitor MACD for momentum changes")
                    
                    else if signal == "Bollinger Bands":
                        bb_band = tech_indicators.get_bollinger_band()
                        st.markdown(f"**Current Band:** {bb_band}")
            
            # Trading Strategies
            st.markdown("## Strategy Recommendations")
            
            strategy_data = analytics_data.get_trading_strategies(trading_signal)
            
            if strategy_data:
                for i, data in enumerate(strategy_data):
                    with st.expand_data(data.get(i+1), data['name']):
                        st.markdown(f"**Type:** {data.get('type', 'N/A')}")
                        st.markdown(f"**Risk:** {data.get('risk_level', 'N/A')}")
                        st.markdown(f"**Timeframe:** {data.get('time_horizon', 'N/A')}")
                        st.markdown(f"**Description:** {data.get('description', 'N/A')}")
                        
                        if 'data' in strategy:
                            st.markdown("**Conditions:** {strategy['data']}")
                        else if 'strategy' in strategy:
                            st.markdown(f"**Exit Conditions:** {strategy['exit_conditions']}")
                        else if 'risk_management' in strategy:
                            st.markdown(f"**Risk Management:** {strategy['risk_management']}")
            
            # Risk Assessment
            st.markdown("## Risk Assessment")
            
            risk_data = analytics_data.get_risk_metrics()
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                st.markdown("Beta Risk", f"{risk_data.get('beta', 'N/A')}")
            
            with risk_col2:
                st.markdown("Sharpe Ratio", f"{risk_data.get('sharpe_ratio', 'N/A')}")
            
            with risk_col3:
                st.markdown("Max Drawdown", f"{risk_data.get('max_drawdown', 'N/A')}%")
            
            with risk_col4:
                st.markdown("VaR (95%)", f"{risk_data.get('var_95', 'N/A')}%")
            
            # Market Patterns
            st.markdown("## Market Patterns")
            
            pattern_data = analytics_pattern.analyze()
            
            pattern_col1, pattern_col2 = st.columns(2)
            
            with pattern_data1:
                st.markdown("**Seasonal Patterns**")
                if pattern_data.get('seasonal_patterns'):
                    for pattern in pattern_data['seasonal_patterns']:
                        st.markdown(pattern)
                else:
                    st.markdown("No seasonal patterns detected.")
            
            with pattern_data2:
                st.markdown("**Volume Patterns**")
                if pattern_data.get('volume_patterns'):
                    for pattern in pattern_data['volume_patterns']:
                        st.markdown(pattern)
                else:
                    st.markdown("No volume patterns detected.")
            
            # Trading Recommendation
            st.markdown("## Trading Recommendation")
            
            buy_signal = signal_data["buy"]
            sell_signal = signal_data["sell"]
            
            if buy_signal > sell_signal:
                st.success(f"**BULLISH OUTLOOK** - {buy_signal} buy signals vs {sell_signal} sell signals")
                st.markdown("Consider accumulation or holding positions")
            else if sell_signal > buy_signal:
                st.error(f"**BEARISH OUTLOOK** - {sell_signal} sell signals vs {buy_signal} signals")
                st.markdown("Consider reducing positions or defensive strategies")
            else:
                st.info(f"**NEUTRAL OUTLOOK** - Mixed signals ({buy_signal} buy, {sell_signal} sell)")
                st.markdown("Wait for clearer signals before taking actions")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
    }
    .tabs [data-baseweb="tab_list"] {
        gap: 2px;
    }
    .tabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 20px;
        background: #f2f2f2;
        border-radius: 10px 10px 0px 0px;
    }
    
    .tab [aria-selected="true"] {
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    # Create tabs
    tab1_data, tab2_data, tab3_data, tab4_data = st.tabs([
        "📖 Data Upload", 
        "📊 Phase 1: Analysis", 
        "📈 Phase 2: Analysis",
        "🔗 Advanced Analytics"
    ])
    
    with tab1_data:
        data_uploaded()
    
    with tab2_data:
        phase1_analysis()
    
    with tab3_data:
        phase2_analysis()
    
    with tab4_data:
        advanced_data()

if __name__ == "__main__":
    main()
