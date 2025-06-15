import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime as dt, timedelta
import warnings
import yfinance as yf

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
from utils.html_report_generator import HTMLReportGenerator

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
    if st.button("🗑️ Clear All Data", type="secondary", help="Refresh page to clear all data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Initialize session state
session_keys = [
    'current_data', 'previous_data', 'historical_data', 'selected_symbol',
    'selected_stock_symbol', 'data_quality_report', 'comparative_analysis', 'yfinance_data'
]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

def data_upload_section():
    st.header("📁 Enhanced Data Upload & Processing")
    st.markdown("Upload your stock data files for comprehensive Phase 1 & Phase 2 analysis")

    if st.button("🗑️ Clear All Uploaded Files", type="secondary", key="clear_all_files"):
        keys_to_clear = [
            'current_data', 'previous_data', 'daily_data', 'historical_data',
            'comparative_analysis', 'yfinance_data', 'data_quality_report',
            'selected_symbol', 'selected_stock_symbol', 'current_data_file',
            'previous_data_file', 'historical_data_file'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Explicitly clear file uploader states
        st.session_state['current_data_file'] = None
        st.session_state['previous_data_file'] = None
        st.session_state['historical_data_file'] = None
        st.success("Successfully cleared all data and uploaded files!")
        st.rerun()

    robust_processor = RobustProcessor()

    st.subheader("📊 Current Stock Data")
    current_file = st.file_uploader(
        "Upload Current Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="current_data_file",
        help="Upload current stock data with columns like Symbol, Name, Last Sale, % Change"
    )

    if current_file:
        with st.spinner("Processing current stock data..."):
            try:
                current_data, quality_report = robust_processor.process_uploaded_data(current_file)
                if current_data is not None:
                    st.session_state.current_data = current_data
                    st.session_state.data_quality_report = quality_report
                    st.success(f"✅ Current data loaded! ({len(current_data)} stocks)")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Stocks", len(current_data))
                    with col2:
                        valid_count = len(current_data[current_data['Symbol'].notna() &
                                                      (current_data['Symbol'].str.strip() != '') &
                                                      (current_data['Symbol'].str.strip() != 'nan')])
                        st.metric("Valid Symbols", valid_count)
                    with col3:
                        st.metric("Data Completeness", f"{quality_report.get('completeness_score', 0):.1f}%")
                    with col4:
                        st.metric("Quality Score", f"{quality_report.get('overall_quality', 0):.1f}/10")
                    with st.expander("📋 Sample Data Preview"):
                        st.dataframe(current_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process current data file.")
            except Exception as e:
                st.error(f"Error processing current data: {str(e)}")

    st.subheader("📈 Previous Stock Data (For Comparative Analysis)")
    previous_file = st.file_uploader(
        "Upload Previous Stock Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="previous_data_file",
        help="Upload previous period stock data for Phase 1"
    )

    if previous_file:
        with st.spinner("Processing previous stock data..."):
            try:
                previous_data, _ = robust_processor.process_uploaded_data(previous_file)
                if previous_data is not None:
                    st.session_state.previous_data = previous_data
                    st.success(f"✅ Previous data loaded! ({len(previous_data)} stocks)")
                    with st.expander("📋 Previous Data Preview"):
                        st.dataframe(previous_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process previous data file.")
            except Exception as e:
                st.error(f"Error processing previous data: {str(e)}")

    st.subheader("📉 Historical Price Data (Optional)")
    historical_file = st.file_uploader(
        "Upload Historical Price Data (Excel/CSV)",
        type=['xlsx', 'xls', 'csv'],
        key="historical_data_file",
        help="Upload historical data with Date, Open, High, Low, Close, Volume"
    )

    if historical_file:
        with st.spinner("Processing historical data..."):
            try:
                processor = DataProcessor()
                historical_data, extracted_symbol = processor.process_historical_data(historical_file)
                if historical_data is not None:
                    # Validate required columns
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in historical_data.columns]
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                    else:
                        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
                        historical_data = historical_data.set_index('Date')
                        st.session_state.historical_data = historical_data
                        if extracted_symbol:
                            st.session_state.selected_symbol = extracted_symbol
                        st.success(f"✅ Historical data loaded! ({len(historical_data)} data points)")
                        with st.expander("📋 Historical Data Preview"):
                            st.dataframe(historical_data.head(), use_container_width=True)
                else:
                    st.error("Failed to process historical data file.")
            except Exception as e:
                st.error(f"Error processing historical data: {str(e)}")

    # Optional: Remove historical data upload to simplify UI
    # To remove, delete the historical_file block above and this comment

def phase1_comparative_analysis_section():
    st.header("📊 Phase 1: Comparative Analysis")
    if st.session_state.current_data is None or st.session_state.previous_data is None:
        st.warning("⚠️ Please upload both current and previous stock data.")
        return

    with st.spinner("Performing comparative analysis..."):
        comp_analysis = EnhancedComparativeAnalysis(st.session_state.current_data, st.session_state.previous_data)
        if comp_analysis.merged_data is None or comp_analysis.merged_data.empty:
            st.error("No matching stocks found.")
            return
        merged_data = comp_analysis.merged_data

    st.subheader("📈 Performance Summary")
    summary = comp_analysis.get_performance_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", summary.get('total_stocks', 0))
    with col2:
        st.metric("Average Change", f"{summary.get('avg_change', 0):.2f}%")
    with col3:
        st.metric("Gainers", summary.get('gainers', 0))
    with col4:
        st.metric("Losers", summary.get('losers', 0))

    st.subheader("📊 Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        symbols = comp_analysis.get_all_symbols()
        search_symbol = st.selectbox("Stock Symbol", ["None"] + symbols, key="filter_search_symbol")
        if search_symbol != "None":
            st.session_state.selected_stock_symbol = search_symbol
    with filter_col2:
        sectors = comp_analysis.get_available_sectors()
        selected_sector = st.selectbox("Sector", ["All"] + sectors, key="filter_sector")
    with filter_col3:
        performance_filter = st.selectbox(
            "Performance",
            ["All Stocks", "Gainers Only", "Losers Only", "Top 10 Performers", "Bottom 10 Performers"],
            key="filter_performance"
        )
    with filter_col4:
        min_change = st.number_input("Min Change %", value=float(merged_data['Price_Change_Pct'].min()), key="filter_min_change")
    filter_col5, filter_col6 = st.columns(2)
    with filter_col5:
        max_change = st.number_input("Max Change %", value=float(merged_data['Price_Change_Pct'].max()), key="filter_max_change")
    with filter_col6:
        if st.button("🧹 Clear Filters", type="secondary", key="clear_filters_phase1_unique"):
            keys_to_clear = [
                'filter_search_symbol', 'filter_sector', 'filter_performance',
                'filter_min_change', 'filter_max_change', 'selected_stock_symbol'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Filters cleared!")
            st.rerun()

    filtered_df = merged_data[(merged_data['Price_Change_Pct'] >= min_change) & (merged_data['Price_Change_Pct'] <= max_change)]
    if search_symbol != "None":
        filtered_df = filtered_df[filtered_df['Symbol'] == search_symbol]
    if selected_sector != "All":
        sector_filtered = comp_analysis.filter_by_sector(selected_sector)
        if not sector_filtered.empty:
            filtered_df = filtered_df[filtered_df['Symbol'].isin(sector_filtered['Symbol'])]
    if performance_filter == "Gainers Only":
        filtered_df = filtered_df[filtered_df['Price_Change_Pct'] > 0]
    elif performance_filter == "Losers Only":
        filtered_df = filtered_df[filtered_df['Price_Change_Pct'] < 0]
    elif performance_filter == "Top 10 Performers":
        filtered_df = filtered_df.nlargest(10, 'Price_Change_Pct')
    elif performance_filter == "Bottom 10 Performers":
        filtered_df = filtered_df.nsmallest(10, 'Price_Change_Pct')

    st.subheader("📈 Filtered Results")
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            label="📥 Download Filtered Data",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_stocks.csv",
            mime="text/csv"
        )

def phase2_deep_analysis_section():
    st.header("📈 Phase 2: Deep Stock Analysis")
    available_stocks = []
    if st.session_state.current_data is not None:
        available_stocks.extend(st.session_state.current_data['Symbol'].dropna().unique().tolist())
    if st.session_state.previous_data is not None:
        available_stocks.extend([s for s in st.session_state.previous_data['Symbol'].dropna().unique() if s not in available_stocks])
    if not available_stocks:
        st.warning("⚠️ Please upload stock data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_stock = st.selectbox("Select Stock", options=available_stocks)
        if selected_stock:
            st.session_state.selected_stock_symbol = selected_stock
    with col2:
        date_option = st.radio("Date Range", ["Predefined Periods", "Custom Range"])
        if date_option == "Predefined Periods":
            analysis_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
            start_date, end_date = None, None
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date", dt.now() - timedelta(days=365))
            with col_end:
                end_date = st.date_input("End Date", dt.now())
            analysis_period = None

    if selected_stock and st.button("🔄 Fetch & Analyze"):
        with st.spinner(f"Fetching data for {selected_stock}..."):
            ticker = yf.Ticker(selected_stock)
            hist_data = ticker.history(period=analysis_period) if analysis_period else ticker.history(start=start_date, end=end_date)
            if hist_data.empty:
                st.error(f"No data for {selected_stock}.")
                return
            st.session_state.yfinance_data = hist_data
            info = ticker.info
            company_name = info.get('longName', selected_stock)
            st.success(f"✅ Analyzing {company_name} ({selected_stock})")
            data_clean = hist_data.reset_index().rename(columns={'Date': 'Datetime'})
            viz = Visualizations(historical_data=data_clean)
            st.plotly_chart(viz.create_candlestick_chart(), use_container_width=True)

def advanced_analytics_section():
    st.header("🔮 Advanced Analytics & Predictions")
    if not st.session_state.selected_stock_symbol:
        st.warning("⚠️ Select a stock in Phase 2.")
        return
    data_source = st.session_state.yfinance_data if st.session_state.yfinance_data is not None else st.session_state.historical_data
    if data_source is None or data_source.empty:
        st.error("No historical data available.")
        return

    # Clean data to avoid Datetime ambiguity
    data_clean = data_source.copy().reset_index()
    if 'Datetime' in data_clean.columns and 'Date' in data_clean.columns:
        data_clean = data_clean.drop(columns=['Datetime'])
    elif 'Datetime' in data_clean.columns:
        data_clean = data_clean.rename(columns={'Datetime': 'Date'})
    data_clean['Date'] = pd.to_datetime(data_clean['Date'])
    data_clean = data_clean.set_index('Date')

    pred_tab, viz_tab, insights_tab = st.tabs(["🔮 Predictions", "📊 Visualizations", "💡 Insights"])

    with pred_tab:
        if len(data_clean) > 50:
            predictions = PricePredictions(data_clean.reset_index())
            pred_days = st.slider("Prediction Days", 1, 30, 7)
            pred_method = st.selectbox("Method", ["technical_analysis", "linear_trend", "moving_average"])
            if st.button("Generate Predictions"):
                pred_prices = predictions.predict_prices(pred_days, pred_method)
                if pred_prices:
                    st.plotly_chart(predictions.create_prediction_chart(pred_prices, pred_days), use_container_width=True)

    with viz_tab:
        viz = Visualizations(historical_data=data_clean.reset_index())
        risk_gauge = RiskGauge()
        try:
            st.plotly_chart(viz.create_candlestick_chart(), use_container_width=True)
            st.plotly_chart(viz.create_price_trends_chart(), use_container_width=True)
            if len(data_clean) > 20:
                returns = data_clean['Close'].pct_change().dropna()
                volatility = returns.std() * 100
                total_return = ((data_clean['Close'].iloc[-1] / data_clean['Close'].iloc[0]) - 1) * 100
                risk_levels = np.linspace(0, 100, 20)
                return_levels = np.linspace(-50, 150, 20)
                X, Y = np.meshgrid(risk_levels, return_levels)
                Z = 100 * np.exp(-((X-volatility)**2 + (Y-total_return)**2) / 2000)
                fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
                fig.update_layout(title='3D Factor Analysis', scene=dict(xaxis_title='Risk (%)', yaxis_title='Return (%)', zaxis_title='Score'))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error generating visualizations: {str(e)}")

    with insights_tab:
        if len(data_clean) > 50:
            tech_indicators = TechnicalIndicators(data_clean.reset_index())
            analytics = Analytics(historical_data=data_clean)
            signals = tech_indicators.get_trading_signals()
            enhanced_signals = {}
            signal_summary = {"buy": 0, "sell": 0, "hold": 0}
            for indicator, signal_data in signals.items():
                signal = signal_data.get('signal', 'Unknown').lower()
                strength = 'Unknown'
                if indicator == 'RSI':
                    rsi = tech_indicators.calculate_rsi().iloc[-1]
                    strength = 'High' if rsi > 70 or rsi < 30 else 'Medium' if rsi > 60 or rsi < 40 else 'Low'
                elif indicator == 'MACD':
                    macd = tech_indicators.calculate_macd()
                    if not macd.empty:
                        macd_diff = macd['MACD'] - macd['Signal']
                        strength = 'High' if abs(macd_diff.iloc[-1]) > macd_diff.std() else 'Medium' if abs(macd_diff.iloc[-1]) > macd_diff.std() / 2 else 'Low'
                enhanced_signals[indicator] = {'signal': signal_data.get('signal'), 'strength': strength}
                if 'buy' in signal:
                    signal_summary["buy"] += 1
                elif 'sell' in signal:
                    signal_summary["sell"] += 1
                else:
                    signal_summary["hold"] += 1
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy Signals", signal_summary["buy"])
            with col2:
                st.metric("Sell Signals", signal_summary["sell"])
            with col3:
                st.metric("Hold Signals", signal_summary["hold"])

    if st.button("🔽 Generate HTML Report"):
        with st.spinner("Generating report..."):
            try:
                report_gen = HTMLReportGenerator()
                stock_symbol = st.session_state.selected_stock_symbol or "STOCK"
                tech_indicators = TechnicalIndicators(data_clean.reset_index())
                analytics = Analytics(historical_data=data_clean)
                viz = Visualizations(historical_data=data_clean.reset_index())
                risk_gauge = RiskGauge()
                visualizations_dict = {
                    'candlestick': viz.create_candlestick_chart(),
                    'price_trends': viz.create_price_trends_chart(),
                    'volume_analysis': viz.create_volume_chart(),
                    '3d_factor_analysis': go.Figure(data=[go.Surface(x=np.linspace(0, 100, 20), y=np.linspace(-50, 150, 20),
                                                                    z=np.random.rand(20, 20), colorscale='Viridis')])
                }
                report_html = report_gen.generate_comprehensive_report(
                    stock_symbol=stock_symbol,
                    historical_data=data_clean,
                    tech_indicators=tech_indicators,
                    analytics=analytics,
                    visualizations=viz,
                    predictions=None,
                    advanced_analytics=st.session_state.comparative_analysis,
                    additional_figures=visualizations_dict
                )
                st.download_button(
                    label="📄 Download Report",
                    data=report_html,
                    file_name=f"{stock_symbol}_report.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def main():
    st.markdown("""
    <style>
    .success-card { background-color: #e6ffed; padding: 15px; border-radius: 8px; }
    .warning-card { background-color: #fff4e6; padding: 15px; border-radius: 8px; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    tabs = st.tabs(["📁 Data Upload", "📊 Phase 1: Comparative", "📈 Phase 2: Deep Analysis", "🔮 Advanced Analytics"])
    with tabs[0]:
        data_upload_section()
    with tabs[1]:
        phase1_comparative_analysis_section()
    with tabs[2]:
        phase2_deep_analysis_section()
    with tabs[3]:
        advanced_analytics_section()

if __name__ == "__main__":
    main()
