import streamlit as st
import pandas as pd
from io import StringIO

def load_csv(file):
    """Load CSV file and return as DataFrame"""
    content = file.getvalue().decode('utf-8')
    return pd.read_csv(StringIO(content))

def clean_numeric(value):
    """Handle currency symbols, commas, etc., to clean and convert them to numeric"""
    try:
        return float(str(value).replace('$', '').replace(',', '').replace('%', '').strip())
    except ValueError:
        return 0  # Return 0 if it's not a valid number

def calculate_profit_loss(df_prev, df_curr):
    """Merge the two dataframes and calculate profit/loss and percentage change"""
    df_merged = pd.merge(df_prev, df_curr, on='Symbol', suffixes=('_prev', '_curr'))
    
    # Clean and convert necessary columns to numeric
    for col in ['Last Sale', 'Net Change', '% Change']:
        df_merged[f'{col}_prev'] = df_merged[f'{col}_prev'].apply(clean_numeric)
        df_merged[f'{col}_curr'] = df_merged[f'{col}_curr'].apply(clean_numeric)
    
    # Calculate Profit/Loss and % Change for display
    df_merged['Profit/Loss'] = df_merged['Last Sale_curr'] - df_merged['Last Sale_prev']
    df_merged['% Change_calc'] = ((df_merged['Profit/Loss'] / df_merged['Last Sale_prev']) * 100).round(2)
    
    return df_merged

def color_profit_loss(val):
    """Color code profit/loss columns (Green for profit, Red for loss)"""
    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    return f'color: {color}'

def color_change_values(val):
    """Color code percentage change columns (Green for positive, Red for negative)"""
    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    return f'color: {color}'

def color_earnings(val):
    """Color code earnings columns (Green for profit, Red for loss)"""
    try:
        val = float(val)
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    except ValueError:
        color = 'black'  # For non-numeric values, keep the color as black
    return f'color: {color}'

st.set_page_config(layout="wide")
st.title("Stock Data Comparison Tool")

# Sidebar for file upload
st.sidebar.header("File Upload & Settings")
previous_file = st.sidebar.file_uploader("Upload previous CSV", type=["csv"])
current_file = st.sidebar.file_uploader("Upload current CSV", type=["csv"])

# Check if both previous and current files are uploaded
if previous_file and current_file:
    df_prev = load_csv(previous_file)
    df_curr = load_csv(current_file)
    
    required_columns = ['Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', 'Industry']
    if all(col in df_prev.columns for col in required_columns) and all(col in df_curr.columns for col in required_columns):
        # Recalculate only when files are uploaded or if session state is empty
        if 'comparison_df' not in st.session_state:
            st.session_state.comparison_df = calculate_profit_loss(df_prev, df_curr)

        comparison_df = st.session_state.comparison_df  # Use stored comparison data

        # Display options for filtering (Positive % Change, Negative % Change, All)
        display_option = st.sidebar.radio(
            "Select display option:",
            ["Display All", "Positive % Change", "Negative % Change"]
        )
        
        # Filters for sector and industry
        sector_filter = st.sidebar.multiselect("Filter by Sector", options=['All'] + list(comparison_df['Sector_curr'].unique()))
        selected_sectors = sector_filter if sector_filter != ['All'] else list(comparison_df['Sector_curr'].unique())
        industries_in_selected_sectors = comparison_df[comparison_df['Sector_curr'].isin(selected_sectors)]['Industry_curr'].unique()
        industry_filter = st.sidebar.multiselect("Filter by Industry", options=['All'] + list(industries_in_selected_sectors))

        # Apply display option
        filtered_df = comparison_df.copy()
        if display_option == "Positive % Change":
            filtered_df = filtered_df[filtered_df['% Change_calc'] > 0]
        elif display_option == "Negative % Change":
            filtered_df = filtered_df[filtered_df['% Change_calc'] < 0]

        # Apply sector and industry filters
        if sector_filter and 'All' not in sector_filter:
            filtered_df = filtered_df[filtered_df['Sector_curr'].isin(sector_filter)]
        if industry_filter and 'All' not in industry_filter:
            filtered_df = filtered_df[filtered_df['Industry_curr'].isin(industry_filter)]

        # Display the filtered stock data
        st.subheader("Stock Data Comparison")
        st.write(f"Number of rows displayed: {len(filtered_df)}")
        
        if filtered_df.empty:
            st.write("No data available after applying filters.")
        else:
            # Remove unwanted columns from the displayed output
            filtered_df = filtered_df.drop(columns=[
                'Name_prev', 'Market Cap_prev', 'IPO Year_prev', 'Sector_prev', 'Industry_prev',
                'Name_curr', 'Market Cap_curr', 'IPO Year_curr', 'Country_prev'
            ])
            
            # Reorder columns to ensure Volume, Sector, Industry, Country come at the end
            filtered_df = filtered_df[['Symbol', 'Last Sale_prev', 'Net Change_prev', '% Change_prev', 
                                       'Last Sale_curr', 'Net Change_curr', '% Change_curr', 
                                       'Profit/Loss', '% Change_calc', 
                                       'Volume_prev', 'Volume_curr', 'Sector_curr', 'Industry_curr', 'Country_curr']]

            # Allow user to choose a column to sort by and sort the data
            sort_column = st.selectbox("Select column to sort by", filtered_df.columns)
            sort_order = st.radio("Select sort order", ['Ascending', 'Descending'])

            # Sort the DataFrame
            ascending = True if sort_order == 'Ascending' else False
            sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

            # Apply color coding for Profit/Loss and percentage changes
            styled_df = sorted_df.style.applymap(color_change_values, subset=['Net Change_prev', '% Change_prev', 'Net Change_curr', '% Change_curr'])
            styled_df = styled_df.applymap(color_profit_loss, subset=['Profit/Loss', '% Change_calc'])
            st.dataframe(styled_df, use_container_width=True)

        # Symbol search feature
        symbol_search = st.selectbox("Search for a specific symbol:", [''] + list(filtered_df['Symbol']))
        if symbol_search:
            st.dataframe(filtered_df[filtered_df['Symbol'] == symbol_search], use_container_width=True)

        # Earnings Data Upload and Comparison
        earnings_check = st.checkbox("Enable Earnings Data Comparison")
        if earnings_check:
            earnings_file = st.file_uploader("Upload Earnings CSV", type=["csv"])

            if earnings_file:
                # Load earnings data
                earnings_df = load_csv(earnings_file)
                earnings_columns = ['Symbol', 'Company', 'Market Cap(M)', 'Time', 'Estimate', 'Reported', 'Surprise', '%Surp', '%Price Change**']

                if all(col in earnings_df.columns for col in earnings_columns):
                    # Merge earnings data with comparison data based on Symbol
                    merged_df = pd.merge(sorted_df, earnings_df, on='Symbol', how='inner', suffixes=('', '_EAR'))

                    # Select columns to display
                    merged_df_display = merged_df[['Symbol', 'Company', 'Last Sale_prev', 'Net Change_prev', '% Change_prev', 'Last Sale_curr', 
                                                   'Net Change_curr', '% Change_curr', 'Profit/Loss', '% Change_calc',
                                                   'Market Cap(M)', 'Time', 'Estimate', 'Reported', 'Surprise', '%Surp', '%Price Change**','Sector_curr','Industry_curr']]
                    merged_df_display.rename(columns={
                        'Symbol': 'EAR_symbol',
                        'Company': 'EAR_company',
                        'Market Cap(M)': 'EAR_Market Cap(M)',
                        'Time': 'EAR_Time',
                        'Estimate': 'EAR_Estimate',
                        'Reported': 'EAR_Reported',
                        'Surprise': 'EAR_Surprise',
                        '%Surp': 'EAR_%Surp',
                        '%Price Change**': 'EAR_%Price Change**'
                    }, inplace=True)

                    # Apply color coding to the earnings data
                    styled_earnings_df = merged_df_display.style.applymap(color_earnings, subset=['EAR_Estimate', 'EAR_Reported', 'EAR_Surprise', 'EAR_%Surp', 'EAR_%Price Change**'])
                    styled_earnings_df = styled_earnings_df.applymap(color_change_values, subset=['Net Change_prev', '% Change_prev', 'Net Change_curr', '% Change_curr', 'Profit/Loss', '% Change_calc'])
                    st.dataframe(styled_earnings_df, use_container_width=True)

                    # Option to download the earnings comparison result
                    csv_earnings = merged_df_display.to_csv(index=False)
                    st.sidebar.download_button(label="Download Earnings Comparison CSV", data=csv_earnings, file_name="earnings_comparison.csv", mime="text/csv")
                else:
                    st.error("The uploaded earnings CSV must contain the required columns.")
            else:
                st.write("Please upload the earnings CSV file.")

    else:
        st.error("The uploaded CSV files must contain the 'Symbol', 'Last Sale', 'Net Change', '% Change', 'Sector', and 'Industry' columns.")
else:
    st.write("Please upload both CSV files to start the comparison.")
