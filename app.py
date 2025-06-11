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
    if ('current_data' in st.session_state and 'previous_data' in st.session_state):
        st.success("🎉 Both datasets are ready for analysis!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.success("✅ Analysis ready! Navigate to Phase 1: Comparative Analysis to see results.")
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
                    
                    # Handle missing Adj Close gracefully
                    if 'Adj Close' not in hist_data_clean.columns:
                        hist_data_clean['Adj Close'] = hist_data_clean['Close']
                        st.info("ℹ️ Using Close price for analysis (Adj Close not available)")
                    
                    # Add missing columns with defaults
                    for col in ['Dividends', 'Stock Splits']:
                        if col not in hist_data_clean.columns:
                            hist_data_clean[col] = 0
                    
                    # Rename Date to Datetime for consistency
                    if 'Date' in hist_data_clean.columns:
                        hist_data_clean = hist_data_clean.rename(columns={'Date': 'Datetime'})
                    
                    # Advanced Visualizations
                    st.subheader("📊 Advanced Price Visualizations")
                    
                    # Create enhanced visualizations
                    if len(hist_data_clean) > 0:
                        viz = Visualizations(historical_data=hist_data_clean)
                        
                        # Candlestick chart
                        candlestick_fig = viz.create_candlestick_chart()
                        st.plotly_chart(candlestick_fig, use_container_width=True, key="phase2_candlestick")
                        
                        # Price trends
                        price_trends_fig = viz.create_price_trends_chart()
                        st.plotly_chart(price_trends_fig, use_container_width=True, key="phase2_price_trends")
                        
                        # Volume analysis
                        volume_fig = viz.create_volume_chart()
                        st.plotly_chart(volume_fig, use_container_width=True, key="phase2_volume")
                    
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
                            rsi_chart = tech_indicators.create_rsi_chart()
                            st.plotly_chart(rsi_chart, use_container_width=True, key="phase2_rsi_chart")
                        
                        with tech_col2:
                            macd_chart = tech_indicators.create_macd_chart()
                            st.plotly_chart(macd_chart, use_container_width=True, key="phase2_macd_chart")
                        
                        # Bollinger Bands
                        bb_chart = tech_indicators.create_bollinger_bands_chart()
                        st.plotly_chart(bb_chart, use_container_width=True, key="phase2_bb_chart")
                        
                        # Trading Signals Dashboard
                        st.subheader("🎯 Trading Signals & Recommendations")
                        signals = tech_indicators.get_trading_signals()
                        
                        # Create signal cards
                        signal_cols = st.columns(min(len(signals), 4))
                        for i, (indicator, signal_data) in enumerate(signals.items()):
                            with signal_cols[i % len(signal_cols)]:
                                signal_value = signal_data.get('signal', 'Unknown')
                                signal_strength = signal_data.get('strength', 'Unknown')
                                
                                # Enhanced signal display
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
                    
                    # Performance Metrics
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
                    
                    # Phase 1 Integration
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
    """Advanced Analytics: Predictions, Visualizations, and Trading Insights."""
    st.header("🔮 Advanced Analytics & Predictions")
    st.markdown("Advanced technical analysis, predictions, and comprehensive trading insights")
    
    if st.session_state.historical_data is None and st.session_state.yfinance_data is None:
        st.warning("⚠️ Please upload historical data or fetch yfinance data in Phase 2 first.")
        return
    
    # Use yfinance data if available, otherwise historical data
    data_source = st.session_state.yfinance_data if st.session_state.yfinance_data is not None else st.session_state.historical_data
    
    if data_source is None or data_source.empty:
        st.error("No historical data available for analysis.")
        return
    
    # Prepare data
    data_clean = data_source.copy()
    if hasattr(data_clean, 'reset_index'):
        data_clean = data_clean.reset_index()
    
    # Handle missing Adj Close
    if 'Adj Close' not in data_clean.columns:
        data_clean['Adj Close'] = data_clean['Close']
    
    # Add missing columns
    for col in ['Dividends', 'Stock Splits']:
        if col not in data_clean.columns:
            data_clean[col] = 0
    
    # Rename Date to Datetime
    if 'Date' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Date': 'Datetime'})
    
    # Create tabs for different analytics
    pred_tab, viz_tab, insights_tab = st.tabs(["🔮 Price Predictions", "📊 Advanced Visualizations", "💡 Trading Insights"])
    
    with pred_tab:
        st.subheader("🔮 Price Predictions")
        
        if len(data_clean) > 50:
            predictions = PricePredictions(data_clean)
            
            # Prediction controls
            col1, col2 = st.columns(2)
            
            with col1:
                pred_days = st.slider("Prediction Days", min_value=1, max_value=30, value=7)
            
            with col2:
                pred_method = st.selectbox(
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
                    pred_prices = predictions.predict_prices(pred_days, pred_method)
                    
                    if pred_prices:
                        # Create prediction chart
                        pred_chart = predictions.create_prediction_chart(pred_prices, pred_days)
                        st.plotly_chart(pred_chart, use_container_width=True, key="predictions_chart")
                        
                        # Prediction metrics
                        confidence = predictions.calculate_prediction_confidence()
                        
                        # Display predicted prices in a table
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
                            st.metric("Confidence Score", f"{confidence.get('score', 0):.1f}/10")
                        
                        with col4:
                            volatility = confidence.get('volatility', 0)
                            st.metric("Prediction Volatility", f"{volatility:.2f}%")
                        
                        # Disclaimer
                        st.info(predictions.get_prediction_disclaimer())
                    else:
                        st.error("Unable to generate predictions. Please try a different method.")
        else:
            st.warning("Insufficient data for predictions (need >50 data points)")
    
    with viz_tab:
        st.subheader("📊 Advanced Visualizations")
        
        # Add comprehensive explanation for 3D Factor Analysis
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
            # Initialize visualizations
            viz = Visualizations(historical_data=data_clean)
            
            # Create visualization options
            viz_option = st.selectbox(
                "Select Visualization",
                ["Candlestick Chart", "Price Trends", "Volume Analysis", "Market Overview Dashboard", "3D Factor Analysis"]
            )
            
            if viz_option == "Candlestick Chart":
                fig = viz.create_candlestick_chart()
                st.plotly_chart(fig, use_container_width=True, key="advanced_candlestick")
            
            elif viz_option == "Price Trends":
                fig = viz.create_price_trends_chart()
                st.plotly_chart(fig, use_container_width=True, key="advanced_price_trends")
            
            elif viz_option == "Volume Analysis":
                fig = viz.create_volume_chart()
                st.plotly_chart(fig, use_container_width=True, key="advanced_volume")
            
            elif viz_option == "3D Factor Analysis":
                st.markdown("**🌐 3D Factor Analysis**")
                st.info("This advanced visualization shows the relationship between Risk, Return, and Market Correlation in a 3D space.")
                
                # Create 3D factor analysis
                risk_gauge = RiskGauge()
                
                if len(data_clean) > 20:
                    # Calculate metrics for 3D analysis
                    returns = data_clean['Close'].pct_change().dropna()
                    volatility = returns.std() * 100
                    total_return = ((data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0]) - 1) * 100
                    
                    # Create sample data for 3D surface
                    risk_levels = np.linspace(0, 100, 20)
                    return_levels = np.linspace(-50, 150, 20)
                    X, Y = np.meshgrid(risk_levels, return_levels)
                    
                    # Generate Z values based on a performance function
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add current stock position indicator
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Risk Level", f"{volatility:.1f}%")
                    with col2:
                        st.metric("Total Return", f"{total_return:.1f}%")
                    with col3:
                        performance_score = min(100, max(0, 50 + total_return/2))
                        st.metric("Performance Score", f"{performance_score:.1f}/100")
                else:
                    st.warning("Need more data points for 3D analysis (minimum 20 days)")
            
            elif viz_option == "Market Overview Dashboard":
                # Create professional risk analysis dashboard
                risk_gauge = RiskGauge()
                
                # Calculate risk metrics from historical data
                if len(data_clean) > 1:
                    price_volatility = data_clean['Close'].pct_change().std() * 100
                    volume_volatility = data_clean['Volume'].pct_change().std() * 100 if 'Volume' in data_clean.columns else 30
                    price_trend = (data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0] - 1) * 100
                    
                    # Create risk metrics
                    risk_data = {
                        'risk_score': min(100, max(0, price_volatility * 2)),
                        'volatility': min(100, price_volatility),
                        'performance': min(100, max(0, price_trend + 50)),
                        'sentiment': 50 + (price_trend * 0.5),
                        'liquidity': min(100, max(20, 100 - volume_volatility))
                    }
                    
                    # Professional Risk Dashboard
                    st.subheader("🎯 Professional Risk Assessment Dashboard")
                    
                    # Risk gauge row
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
                    
                    # Advanced Multi-Gauge Dashboard
                    st.subheader("📊 Advanced Multi-Metric Dashboard")
                    advanced_fig = risk_gauge.create_advanced_dashboard(risk_data)
                    st.plotly_chart(advanced_fig, use_container_width=True, key="advanced_dashboard")
                    
                    # 3D Risk Surface Analysis
                    st.subheader("🌐 3D Risk Surface Analysis")
                    surface_fig = risk_gauge.create_3d_surface_plot(data_clean)
                    st.plotly_chart(surface_fig, use_container_width=True, key="3d_surface")
                    
                    # 3D Factor Graph Explanation
                    with st.expander("📚 How to Read the 3D Factor Graph - Real Example", expanded=False):
                        st.markdown("""
                        ### Understanding the 3D Risk Surface
                        
                        **What it shows:** The 3D surface represents the relationship between three key financial factors:
                        - **X-axis (Price Volatility):** How much the stock price fluctuates daily
                        - **Y-axis (Volume Volatility):** How much trading volume changes daily  
                        - **Z-axis (Risk Score):** Combined risk level from both factors
                        
                        **Real-time Example:**
                        
                        **Scenario 1 - Tech Stock (High Volatility)**
                        - Price moves 5% daily (high volatility)
                        - Volume varies 200% daily (high volume volatility)
                        - **Result:** High peak on 3D surface = High Risk (red zone)
                        
                        **Scenario 2 - Utility Stock (Low Volatility)**
                        - Price moves 1% daily (low volatility)
                        - Volume varies 20% daily (stable volume)
                        - **Result:** Low valley on 3D surface = Low Risk (green zone)
                        
                        **How to Use This:**
                        - **Green valleys** = Safer investments with stable price and volume
                        - **Red peaks** = Higher risk investments with unpredictable patterns
                        - **Yellow slopes** = Moderate risk requiring careful monitoring
                        
                        **Trading Decision:**
                        If you see your stock in a red peak area, consider:
                        - Reducing position size
                        - Setting tighter stop losses
                        - Waiting for movement to green/yellow zones
                        """)
                        
                        # Current stock position explanation
                        current_vol = price_volatility
                        current_risk = risk_data['risk_score']
                        
                        if current_risk < 30:
                            risk_zone = "Green Zone (Low Risk)"
                            advice = "Stable investment suitable for conservative portfolios"
                        elif current_risk < 60:
                            risk_zone = "Yellow Zone (Moderate Risk)"
                            advice = "Monitor closely, good for balanced portfolios"
                        else:
                            risk_zone = "Red Zone (High Risk)"
                            advice = "High volatility - suitable only for aggressive trading"
                        
                        st.info(f"**Current Stock Position:** {risk_zone} - {advice}")
                    
                    # Advanced Correlation Heatmap
                    st.subheader("🔥 Advanced Correlation Matrix")
                    heatmap_fig = risk_gauge.create_heatmap_correlation(data_clean)
                    st.plotly_chart(heatmap_fig, use_container_width=True, key="correlation_heatmap")
                    
                    # Professional Candlestick with Technical Analysis
                    if 'Date' not in data_clean.columns:
                        data_clean = data_clean.reset_index()
                        if 'Date' not in data_clean.columns:
                            data_clean['Date'] = data_clean.index
                    
                    st.subheader("📈 Professional Technical Analysis Chart")
                    candlestick_fig = risk_gauge.create_advanced_candlestick(data_clean)
                    st.plotly_chart(candlestick_fig, use_container_width=True, key="advanced_candlestick_tech")
                else:
                    st.warning("Insufficient data for advanced risk analysis")
            
            # Daily data visualizations if available
            if st.session_state.current_data is not None:
                st.subheader("📈 Market Analysis Visualizations")
                
                daily_viz = Visualizations(daily_data=st.session_state.current_data)
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    market_cap_fig = daily_viz.create_market_cap_chart()
                    st.plotly_chart(market_cap_fig, use_container_width=True, key="advanced_market_cap")
                
                with viz_col2:
                    sector_fig = daily_viz.create_sector_pie_chart()
                    st.plotly_chart(sector_fig, use_container_width=True, key="advanced_sector_pie")
                
                # Correlation heatmap
                corr_fig = daily_viz.create_correlation_heatmap()
                st.plotly_chart(corr_fig, use_container_width=True, key="advanced_correlation")
                
                # Performance scatter
                perf_fig = daily_viz.create_performance_volume_scatter()
                st.plotly_chart(perf_fig, use_container_width=True, key="advanced_performance_scatter")
    
    with insights_tab:
        st.subheader("💡 Enhanced Trading Insights")
        
        if len(data_clean) > 50:
            # Technical Analysis
            tech_indicators = TechnicalIndicators(data_clean)
            trading_signals = tech_indicators.get_trading_signals()
            
            # Analytics
            analytics = Analytics(historical_data=data_clean)
            
            # Trading Signals Overview
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
            
            # Detailed Signal Analysis
            st.subheader("📋 Detailed Signal Analysis")
            
            for indicator, signal_data in trading_signals.items():
                signal_value = signal_data.get('signal', 'Unknown')
                strength = signal_data.get('strength', 'Unknown')
                
                with st.expander(f"{indicator} - {signal_value}"):
                    st.write(f"**Current Signal:** {signal_value}")
                    st.write(f"**Signal Strength:** {strength}")
                    
                    # Add specific recommendations based on indicator
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
            
            # Trading Strategies
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
            
            # Risk Assessment
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
            
            # Market Patterns
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
            
            # Overall Recommendation
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

def main():
    # Custom CSS for beautiful UI
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Premium Financial Analysis Dashboard")
    st.markdown("### Comprehensive Stock Trading Analysis with Advanced Technical Indicators & Predictions")
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload", 
        "📊 Phase 1: Comparative Analysis", 
        "📈 Phase 2: Deep Stock Analysis",
        "🔮 Advanced Analytics"
    ])
    
    with tab1:
        data_upload_section()
    
    with tab2:
        phase1_comparative_analysis_section()
    
    with tab3:
        phase2_deep_analysis_section()
    
    with tab4:
        advanced_analytics_section()

if __name__ == "__main__":
    main()