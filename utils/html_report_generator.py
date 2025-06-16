import streamlit as st
import pandas as pd
import numpy as np
from generate_report import HTMLReportGenerator  # Import from generate_report.py
import os
import zipfile
from datetime import datetime, timedelta

# Mock classes for tech_indicators, analytics, and predictions
class MockTechIndicators:
    def get_trading_signals(self):
        return {
            "RSI": {"signal": "Buy", "strength": "Strong"},
            "MACD": {"signal": "Sell", "strength": "Moderate"},
        }
    def calculate_rsi(self):
        return pd.Series([50, 60, 70], index=pd.date_range("2025-06-01", periods=3))

class MockAnalytics:
    def generate_trading_strategies(self, signals):
        return [{"name": "Trend Following", "type": "Long", "risk_level": "Medium", "time_horizon": "1 month", "description": "Follow the trend."}]
    def calculate_risk_metrics(self):
        return {"sharpe_ratio": 1.2, "max_drawdown": 0.15}
    def analyze_patterns(self):
        return {"seasonal_patterns": ["Strong in Q4"], "volume_patterns": ["High on earnings"]}
    def get_performance_summary(self):
        return {"total_stocks": 100, "avg_change": 5.2, "gainers": 60, "losers": 40}

class MockPredictions:
    def predict_prices(self, days, method):
        return [100 + i for i in range(days)]
    def calculate_prediction_confidence(self):
        return {"score": 8.5, "volatility": 2.3}

st.title("Financial Report Generator")
stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")

# Generate sample historical data
dates = pd.date_range(start="2024-06-01", end="2025-06-15", freq="D")
historical_data = pd.DataFrame({
    "Close": np.random.normal(100, 10, len(dates)),
    "Open": np.random.normal(100, 10, len(dates)),
    "High": np.random.normal(105, 10, len(dates)),
    "Low": np.random.normal(95, 10, len(dates)),
    "Volume": np.random.randint(1000000, 5000000, len(dates))
}, index=dates)

# Mock additional figures (empty for simplicity)
additional_figures = {}

# Initialize mock objects
tech_indicators = MockTechIndicators()
analytics = MockAnalytics()
predictions = MockPredictions()

if st.button("Generate Report"):
    generator = HTMLReportGenerator(output_dir="reports")
    try:
        # Input validation
        if historical_data is None or historical_data.empty:
            st.error("Historical data is missing or empty.")
            raise ValueError("Historical data is missing or empty.")
        if not stock_symbol:
            st.error("Stock symbol is required.")
            raise ValueError("Stock symbol is required.")
        
        report_filename = generator.generate_comprehensive_report(
            stock_symbol=stock_symbol,
            historical_data=historical_data,
            tech_indicators=tech_indicators,
            analytics=analytics,
            visualizations=None,
            predictions=predictions,
            advanced_analytics=analytics,
            additional_figures=additional_figures,
            report_type="full"
        )
        st.success(f"Report generated: {report_filename}")
        
        # Download HTML file
        with open(report_filename, "rb") as f:
            st.download_button(
                label="Download HTML Report",
                data=f,
                file_name=os.path.basename(report_filename),
                mime="text/html"
            )
        
        # Zip and download JSON files (if any)
        zip_filename = report_filename.replace(".html", ".zip")
        json_files = [f for f in os.listdir("reports") if f.endswith(".json")]
        if json_files:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in json_files:
                    zipf.write(os.path.join("reports", file), file)
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="Download Chart Data (ZIP)",
                    data=f,
                    file_name=os.path.basename(zip_filename),
                    mime="application/zip"
                )
        else:
            st.info("No chart data generated for this report.")
        
        # Display log file content
        if os.path.exists("report_generator.log"):
            with open("report_generator.log", "r") as f:
                log_content = f.read()
            st.text_area("Debug Log", log_content, height=300)
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        if os.path.exists("report_generator.log"):
            with open("report_generator.log", "r") as f:
                log_content = f.read()
            st.text_area("Debug Log", log_content, height=300)
