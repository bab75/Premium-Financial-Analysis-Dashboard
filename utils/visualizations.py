import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional
import os
import uuid

class Visualizations:
    """Create interactive visualizations for financial data analysis."""
    
    def __init__(self, daily_data: Optional[pd.DataFrame] = None, historical_data: Optional[pd.DataFrame] = None, output_dir: str = "charts"):
        self.daily_data = daily_data
        self.historical_data = historical_data
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Color schemes for financial charts
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#78909c',
            'volume': '#42a5f5',
            'sectors': px.colors.qualitative.Set3
        }
    
    def save_chart_as_html(self, fig: go.Figure, filename: str) -> None:
        """Save a Plotly figure as an interactive HTML file."""
        if fig is None or len(fig.data) == 0:
            print(f"Cannot save chart '{filename}': Figure is empty or invalid.")
            return
        
        # Ensure the filename ends with .html
        if not filename.endswith('.html'):
            filename += '.html'
        
        # Construct full file path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the figure as HTML
        try:
            fig.write_html(filepath, include_plotlyjs='cdn')
            print(f"Chart saved successfully as '{filepath}'")
        except Exception as e:
            print(f"Error saving chart '{filename}': {str(e)}")

    def create_market_cap_chart(self) -> go.Figure:
        """Create bar chart of top 10 stocks by market cap."""
        if self.daily_data is None or self.daily_data.empty:
            return go.Figure()
        
        # Ensure required columns exist
        required_cols = ['Symbol', 'Market Cap', 'Sector']
        if not all(col in self.daily_data.columns for col in required_cols):
            return go.Figure()
        
        # Clean Market Cap data
        top_10 = self.daily_data.dropna(subset=['Market Cap'])
        top_10 = top_10[top_10['Market Cap'] > 0].nlargest(10, 'Market Cap')
        if top_10.empty:
            return go.Figure()
        
        # Create color map for sectors
        unique_sectors = top_10['Sector'].unique()
        color_map = {sector: self.colors['sectors'][i % len(self.colors['sectors'])] 
                     for i, sector in enumerate(unique_sectors)}
        
        colors = [color_map[sector] for sector in top_10['Sector']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_10['Symbol'],
            y=top_10['Market Cap'],
            text=[f"${val/1e9:.1f}B" for val in top_10['Market Cap']],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>' +
                          'Market Cap: $%{y:,.0f}<br>' +
                          'Sector: %{customdata}<br>' +
                          '<extra></extra>',
            customdata=top_10['Sector']
        ))
        
        fig.update_layout(
            title='Top 10 Stocks by Market Capitalization',
            xaxis_title='Stock Symbol',
            yaxis_title='Market Cap ($)',
            showlegend=False,
            yaxis_tickformat='$,.0f'
        )
        
        return fig
    
    def create_and_save_market_cap_chart(self, filename: str = "market_cap_chart.html") -> go.Figure:
        """Create and save market cap chart as HTML."""
        fig = self.create_market_cap_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_sector_pie_chart(self) -> go.Figure:
        """Create pie chart showing sector distribution."""
        if self.daily_data is None or self.daily_data.empty or 'Sector' not in self.daily_data.columns:
            return go.Figure()
        
        sector_counts = self.daily_data['Sector'].dropna().value_counts()
        if sector_counts.empty:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto',
                marker_colors=self.colors['sectors'][:len(sector_counts)]
            )
        ])
        
        fig.update_layout(
            title='Distribution of Stocks by Sector',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_and_save_sector_pie_chart(self, filename: str = "sector_pie_chart.html") -> go.Figure:
        """Create and save sector pie chart as HTML."""
        fig = self.create_sector_pie_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap for numerical columns."""
        if self.daily_data is None or self.daily_data.empty:
            return go.Figure()
        
        # Select numerical columns
        numerical_cols = ['Last Sale', 'Net Change', '% Change', 'Market Cap', 'Volume']
        available_cols = [col for col in numerical_cols if col in self.daily_data.columns]
        
        if len(available_cols) < 2:
            return go.Figure()
        
        # Clean data for correlation
        corr_data = self.daily_data[available_cols].dropna()
        if corr_data.empty:
            return go.Figure()
        
        corr_matrix = corr_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Financial Metrics',
            width=600,
            height=500
        )
        
        return fig
    
    def create_and_save_correlation_heatmap(self, filename: str = "correlation_heatmap.html") -> go.Figure:
        """Create and save correlation heatmap as HTML."""
        fig = self.create_correlation_heatmap()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_performance_volume_scatter(self) -> go.Figure:
        """Create scatter plot of % Change vs Volume."""
        if self.daily_data is None or self.daily_data.empty:
            return go.Figure()
        
        # Ensure required columns exist
        required_cols = ['Volume', '% Change', 'Industry', 'Symbol', 'Market Cap']
        if not all(col in self.daily_data.columns for col in required_cols):
            return go.Figure()
        
        # Clean data: remove NaN, negative, or invalid Market Cap values
        valid_data = self.daily_data.dropna(subset=['Volume', '% Change', 'Market Cap'])
        valid_data = valid_data[valid_data['Market Cap'] > 0]
        if valid_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create color map for industries (top 10 most common)
        top_industries = valid_data['Industry'].value_counts().head(10).index
        color_map = {industry: self.colors['sectors'][i % len(self.colors['sectors'])] 
                     for i, industry in enumerate(top_industries)}
        
        for industry in top_industries:
            industry_data = valid_data[valid_data['Industry'] == industry]
            
            # Calculate marker sizes with safety checks
            marker_sizes = np.sqrt(industry_data['Market Cap']) / 1e5
            marker_sizes = np.clip(marker_sizes, 5, 50)  # Ensure sizes are between 5 and 50
            
            fig.add_trace(go.Scatter(
                x=industry_data['Volume'],
                y=industry_data['% Change'],
                mode='markers',
                name=industry,
                marker=dict(
                    size=marker_sizes,
                    color=color_map[industry],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                              'Volume: %{x:,.0f}<br>' +
                              'Change: %{y:.2f}%<br>' +
                              'Market Cap: $%{customdata[1]:,.0f}<br>' +
                              'Industry: %{customdata[2]}<br>' +
                              '<extra></extra>',
                customdata=np.column_stack((
                    industry_data['Symbol'],
                    industry_data['Market Cap'],
                    industry_data['Industry']
                ))
            ))
        
        # Add remaining industries as "Other"
        other_data = valid_data[~valid_data['Industry'].isin(top_industries)]
        if not other_data.empty:
            # Calculate marker sizes for "Other" category
            marker_sizes = np.sqrt(other_data['Market Cap']) / 1e5
            marker_sizes = np.clip(marker_sizes, 5, 50)  # Ensure sizes are between 5 and 50
            
            fig.add_trace(go.Scatter(
                x=other_data['Volume'],
                y=other_data['% Change'],
                mode='markers',
                name='Other Industries',
                marker=dict(
                    size=marker_sizes,
                    color='lightgray',
                    opacity=0.5,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                              'Volume: %{x:,.0f}<br>' +
                              'Change: %{y:.2f}%<br>' +
                              'Market Cap: $%{customdata[1]:,.0f}<br>' +
                              'Industry: %{customdata[2]}<br>' +
                              '<extra></extra>',
                customdata=np.column_stack((
                    other_data['Symbol'],
                    other_data['Market Cap'],
                    other_data['Industry']
                ))
            ))
        
        fig.update_layout(
            title='Performance vs Volume Analysis (Bubble size = Market Cap)',
            xaxis_title='Trading Volume',
            yaxis_title='% Change',
            xaxis_type='log',
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_and_save_performance_volume_scatter(self, filename: str = "performance_volume_scatter.html") -> go.Figure:
        """Create and save performance vs volume scatter plot as HTML."""
        fig = self.create_performance_volume_scatter()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_candlestick_chart(self) -> go.Figure:
        """Create candlestick chart for historical data."""
        if self.historical_data is None or self.historical_data.empty:
            return go.Figure()
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.historical_data.columns for col in required_cols):
            return go.Figure()
        
        # Ensure index is datetime
        data = self.historical_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
        
        # Clean data
        data = data.dropna(subset=required_cols)
        if data.empty:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish'],
                name='Price',
                hoverinfo='x+y+name'
            )
        ])
        
        fig.update_layout(
            title='Candlestick Chart',
            xaxis_title='Time Period',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis=dict(
                type='date',
                showticklabels=False,  # Hide date labels from axis
                showgrid=True,
                hoverformat='%m-%d-%y'  # Show MM-DD-YY format on hover
            )
        )
        
        return fig
    
    def create_and_save_candlestick_chart(self, filename: str = "candlestick_chart.html") -> go.Figure:
        """Create and save candlestick chart as HTML."""
        fig = self.create_candlestick_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_price_trends_chart(self) -> go.Figure:
        """Create price trends chart showing Close and Adj Close."""
        if self.historical_data is None or self.historical_data.empty or 'Close' not in self.historical_data.columns:
            return go.Figure()
        
        # Ensure index is datetime
        data = self.historical_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
        
        # Clean data
        data = data.dropna(subset=['Close'])
        if data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ))
        
        if 'Adj Close' in data.columns and data['Adj Close'].notna().any():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Adj Close'],
                mode='lines',
                name='Adjusted Close',
                line=dict(color=self.colors['bearish'], width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                              'Adj Close: $%{y:.2f}<br>' +
                              '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Price Trends Over Time',
            xaxis_title='Time Period',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(
                type='date',
                showticklabels=False,  # Hide date labels from axis
                showgrid=True,
                hoverformat='%m-%d-%y'  # Show MM-DD-YY format on hover
            )
        )
        
        return fig
    
    def create_and_save_price_trends_chart(self, filename: str = "price_trends_chart.html") -> go.Figure:
        """Create and save price trends chart as HTML."""
        fig = self.create_price_trends_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_volume_chart(self) -> go.Figure:
        """Create volume analysis chart."""
        if self.historical_data is None or self.historical_data.empty or 'Volume' not in self.historical_data.columns or 'Close' not in self.historical_data.columns:
            return go.Figure()
        
        # Ensure index is datetime
        data = self.historical_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
        
        # Clean data
        data = data.dropna(subset=['Volume', 'Close'])
        if data.empty:
            return go.Figure()
        
        # Calculate volume moving average
        vol_ma = data['Volume'].rolling(window=20, min_periods=1).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=['Price', 'Volume']
        )
        
        # Price chart
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.colors['bullish'], width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Close: $%{y:.2f}<br>' +
                          '<extra></extra>'
        ), row=1, col=1)
        
        # Volume bars
        colors = []
        for i in range(len(data)):
            if i == 0:
                colors.append(self.colors['neutral'])
            else:
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Volume: %{y:,.0f}<br>' +
                          '<extra></extra>'
        ), row=2, col=1)
        
        # Volume moving average
        fig.add_trace(go.Scatter(
            x=data.index,
            y=vol_ma,
            mode='lines',
            name='Volume MA(20)',
            line=dict(color='orange', width=2),
            hovertemplate='<b>Date:</b> %{x|%m-%d-%y}<br>' +
                          'Volume MA(20): %{y:,.0f}<br>' +
                          '<extra></extra>'
        ), row=2, col=1)
        
        fig.update_layout(
            title='Price and Volume Analysis',
            hovermode='x unified',
            showlegend=True,
            xaxis2=dict(
                type='date',
                showticklabels=False,  # Hide date labels from axis
                showgrid=True,
                hoverformat='%m-%d-%y'  # Show MM-DD-YY format on hover
            )
        )
        
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_xaxes(title_text='Time Period', row=2, col=1)
        
        return fig
    
    def create_and_save_volume_chart(self, filename: str = "volume_chart.html") -> go.Figure:
        """Create and save volume analysis chart as HTML."""
        fig = self.create_volume_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_sector_performance_chart(self) -> go.Figure:
        """Create sector performance comparison chart."""
        if self.daily_data is None or self.daily_data.empty or 'Sector' not in self.daily_data.columns:
            return go.Figure()
        
        # Calculate sector performance metrics
        sector_stats = self.daily_data.dropna(subset=['% Change', 'Market Cap']).groupby('Sector').agg({
            '% Change': ['mean', 'std'],
            'Volume': 'mean',
            'Market Cap': 'mean'
        }).round(2)
        
        if sector_stats.empty:
            return go.Figure()
        
        sector_stats.columns = ['Avg_Change', 'Volatility', 'Avg_Volume', 'Avg_Market_Cap']
        sector_stats = sector_stats.reset_index()
        
        fig = go.Figure()
        
        # Calculate marker sizes with safety checks
        marker_sizes = np.sqrt(sector_stats['Avg_Market_Cap']) / 1e6
        marker_sizes = np.clip(marker_sizes, 5, 50)
        
        fig.add_trace(go.Scatter(
            x=sector_stats['Volatility'],
            y=sector_stats['Avg_Change'],
            mode='markers+text',
            text=sector_stats['Sector'],
            textposition='top center',
            marker=dict(
                size=marker_sizes,
                color=sector_stats['Avg_Change'],
                colorscale='RdYlGn',
                colorbar=dict(title="Avg % Change"),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>' +
                          'Avg Change: %{y:.2f}%<br>' +
                          'Volatility: %{x:.2f}%<br>' +
                          'Avg Market Cap: $%{customdata:,.0f}<br>' +
                          '<extra></extra>',
            customdata=sector_stats['Avg_Market_Cap']
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=sector_stats['Volatility'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Sector Performance vs Volatility (Bubble size = Avg Market Cap)',
            xaxis_title='Volatility (Std Dev of % Change)',
            yaxis_title='Average % Change',
            hovermode='closest'
        )
        
        return fig
    
    def create_and_save_sector_performance_chart(self, filename: str = "sector_performance_chart.html") -> go.Figure:
        """Create and save sector performance chart as HTML."""
        fig = self.create_sector_performance_chart()
        self.save_chart_as_html(fig, filename)
        return fig

    def create_market_overview_dashboard(self) -> go.Figure:
        """Create a comprehensive market overview dashboard."""
        if self.daily_data is None or self.daily_data.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Market Cap Distribution', 'Sector Performance', 
                            'Volume vs Change', 'Country Distribution'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Market Cap Distribution (Pie)
        if 'Market Cap' in self.daily_data.columns:
            market_cap_data = self.daily_data[self.daily_data['Market Cap'] > 0].dropna(subset=['Market Cap'])
            if not market_cap_data.empty:
                market_cap_bins = pd.cut(market_cap_data['Market Cap'], 
                                        bins=[0, 1e9, 10e9, 50e9, float('inf')],
                                        labels=['Small (<$1B)', 'Mid ($1B-$10B)', 
                                                'Large ($10B-$50B)', 'Mega (>$50B)'])
                market_cap_dist = market_cap_bins.value_counts()
                
                fig.add_trace(go.Pie(
                    labels=market_cap_dist.index,
                    values=market_cap_dist.values,
                    name="Market Cap"
                ), row=1, col=1)
        
        # Sector Performance (Bar)
        if 'Sector' in self.daily_data.columns and '% Change' in self.daily_data.columns:
            sector_perf = self.daily_data.dropna(subset=['% Change']).groupby('Sector')['% Change'].mean().sort_values(ascending=True)
            if not sector_perf.empty:
                fig.add_trace(go.Bar(
                    y=sector_perf.index,
                    x=sector_perf.values,
                    orientation='h',
                    name="Sector Performance",
                    marker_color=['green' if x > 0 else 'red' for x in sector_perf.values]
                ), row=1, col=2)
        
        # Volume vs Change (Scatter)
        if 'Volume' in self.daily_data.columns and '% Change' in self.daily_data.columns:
            valid_scatter_data = self.daily_data.dropna(subset=['Volume', '% Change'])
            if not valid_scatter_data.empty:
                fig.add_trace(go.Scatter(
                    x=valid_scatter_data['Volume'],
                    y=valid_scatter_data['% Change'],
                    mode='markers',
                    name="Volume vs Change",
                    marker_color=valid_scatter_data['% Change'],
                    marker_colorscale='RdYlGn'
                ), row=2, col=1)
        
        # Country Distribution (Bar)
        if 'Country' in self.daily_data.columns:
            country_dist = self.daily_data['Country'].dropna().value_counts().head(10)
            if not country_dist.empty:
                fig.add_trace(go.Bar(
                    x=country_dist.index,
                    y=country_dist.values,
                    name="Country Distribution"
                ), row=2, col=2)
        
        fig.update_layout(
            title_text="Market Overview Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_and_save_market_overview_dashboard(self, filename: str = "market_overview_dashboard.html") -> go.Figure:
        """Create and save market overview dashboard as HTML."""
        fig = self.create_market_overview_dashboard()
        self.save_chart_as_html(fig, filename)
        return fig
