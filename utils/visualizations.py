import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

class Visualizations:
    """Create interactive visualizations for financial data analysis."""
    
    def __init__(self, daily_data: Optional[pd.DataFrame] = None, historical_data: Optional[pd.DataFrame] = None):
        self.daily_data = daily_data
        self.historical_data = historical_data
        
        # Color schemes for financial charts
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'neutral': '#78909c',
            'volume': '#42a5f5',
            'sectors': px.colors.qualitative.Set3
        }
    
    def create_market_cap_chart(self) -> go.Figure:
        """Create bar chart of top 10 stocks by market cap."""
        if self.daily_data is None:
            return go.Figure()
        
        # Get top 10 by market cap
        top_10 = self.daily_data.nlargest(10, 'Market Cap')
        
        fig = go.Figure()
        
        # Create color map for sectors
        unique_sectors = top_10['Sector'].unique()
        color_map = {sector: self.colors['sectors'][i % len(self.colors['sectors'])] 
                     for i, sector in enumerate(unique_sectors)}
        
        colors = [color_map[sector] for sector in top_10['Sector']]
        
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
    
    def create_sector_pie_chart(self) -> go.Figure:
        """Create pie chart showing sector distribution."""
        if self.daily_data is None:
            return go.Figure()
        
        sector_counts = self.daily_data['Sector'].value_counts()
        
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
    
    def create_correlation_heatmap(self) -> go.Figure:
        """Create correlation heatmap for numerical columns."""
        if self.daily_data is None:
            return go.Figure()
        
        # Select numerical columns
        numerical_cols = ['Last Sale', 'Net Change', '% Change', 'Market Cap', 'Volume']
        available_cols = [col for col in numerical_cols if col in self.daily_data.columns]
        
        if len(available_cols) < 2:
            return go.Figure()
        
        corr_matrix = self.daily_data[available_cols].corr()
        
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
    
    def create_performance_volume_scatter(self) -> go.Figure:
        """Create scatter plot of % Change vs Volume."""
        if self.daily_data is None:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create color map for industries (top 10 most common)
        top_industries = self.daily_data['Industry'].value_counts().head(10).index
        color_map = {industry: self.colors['sectors'][i % len(self.colors['sectors'])] 
                     for i, industry in enumerate(top_industries)}
        
        for industry in top_industries:
            industry_data = self.daily_data[self.daily_data['Industry'] == industry]
            
            fig.add_trace(go.Scatter(
                x=industry_data['Volume'],
                y=industry_data['% Change'],
                mode='markers',
                name=industry,
                marker=dict(
                    size=np.sqrt(industry_data['Market Cap']) / 1e5,  # Size by market cap
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
        other_data = self.daily_data[~self.daily_data['Industry'].isin(top_industries)]
        if len(other_data) > 0:
            fig.add_trace(go.Scatter(
                x=other_data['Volume'],
                y=other_data['% Change'],
                mode='markers',
                name='Other Industries',
                marker=dict(
                    size=np.sqrt(other_data['Market Cap']) / 1e5,
                    color='black',
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
            title='Performance vs Volume Analysis',
            xaxis_title='Trading Volume',
            yaxis_title='% Volatility',
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
    
    def create_candlestick_chart(self) -> go.Figure:
        """Create candlestick chart for historical data."""
        if self.historical_data is None:
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
        
        try:
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
                    hoverformat='%Y-%m-%d'  # Show YYYY-MM-DD format on hover
                )
            )
        
            return fig
        except Exception as e:
            print(f"Error creating candlestick chart: {str(e)}")
            return go.Figure()
    
    def create_price_trends(self) -> go.Figure:
        """Create price trend chart for historical data."""
        if self.historical_data is None:
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
        
        fig = go.Figure()
        
        try:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['bullish'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              'Close: $%{y:.2f}<br>' +
                              '<extra></extra>'
            ))
        
            if 'Adj Close' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Adj Close'],
                    mode='lines',
                    name='Normalized Close',
                    line=dict(color=self.colors['bearish'], width=2, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                  'Normalized Close: $%{y}<br>' +
                                  '<extra></extra>'
                ))
        
            fig.update_layout(
                title='Price Trend Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(
                    type='date',
                    showticklabels=False,  # Hide date labels from axis
                    showgrid=True,
                    hoverformat='%Y-%m-%d'  # Show YYYY-MM-DD format on hover
                )
            )
        
            return fig
        except Exception as e:
            print(f"Error creating price trends chart: {str(e)}")
            return go.Figure()
    
    def create_volume_chart(self) -> go.Figure:
        """Create volume analysis chart."""
        if self.historical_data is None:
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
        
        # Calculate volume moving average
        vol_ma = data['Volume'].rolling(window=20, min_periods=1).mean()
        
        try:
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
                hovertemplate='<b>Date:</b> %{x}<br>' +
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
                hovertemplate='<b>Date:</b> %{x}<br>' +
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
                hovertemplate='<b>Date:</b> %{x}<br>' +
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
                    hoverformat='%Y-%m-%d'  # Show YYYY-MM-DD format on hover
                )
            )
        
            fig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)
            fig.update_xaxes(title_text='Time Period', row=2, col=1)
        
            return fig
        except Exception as e:
            print(f"Error creating volume chart: {str(e)}")
            return go.Figure()
    
    def create_sector_performance_chart(self) -> go.Figure:
        """Create sector performance comparison chart."""
        if self.daily_data is None:
            return go.Figure()
        
        # Calculate sector performance metrics
        sector_stats = self.daily_data.groupby('Sector').agg({
            '% Change': ['mean', 'std'],
            'Volume': 'mean',
            'Market Cap': 'mean'
        }).round(2)
        
        sector_stats.columns = ['Avg_Change', 'Volatility', 'Avg_Volume', 'Avg_Market_Cap']
        sector_stats = sector_stats.reset_index()
        
        try:
            fig = go.Figure()
        
            fig.add_trace(go.Scatter(
                x=sector_stats['Volatility'],
                y=sector_stats['Avg_Change'],
                mode='markers+text',
                text=sector_stats['Sector'],
                textposition='top center',
                marker=dict(
                    size=np.sqrt(sector_stats['Avg_Market_Cap']) / 1e6,
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
                title='Sector Performance vs Volatility',
                xaxis_title='Volatility (Std Dev of % Change)',
                yaxis_title='Average % Change',
                hovermode='closest'
            )
        
            return fig
        except Exception as e:
            print(f"Error creating sector performance chart: {str(e)}")
            return go.Figure()
    
    def create_market_overview_dashboard(self) -> go.Figure:
        """Create a comprehensive market overview dashboard."""
        if self.daily_data is None:
            return go.Figure()
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Market Cap Distribution', 'Sector Performance', 
                                'Volume vs Change', 'Country Distribution'],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
        
            # Market Cap Distribution (Pie)
            market_cap_bins = pd.cut(self.daily_data['Market Cap'], 
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
            sector_perf = self.daily_data.groupby('Sector')['% Change'].mean().sort_values(ascending=True)
        
            fig.add_trace(go.Bar(
                y=sector_perf.index,
                x=sector_perf.values,
                orientation='h',
                name="Sector Performance",
                marker_color=['green' if x > 0 else 'red' for x in sector_perf.values]
            ), row=1, col=2)
        
            # Volume vs Change (Scatter)
            fig.add_trace(go.Scatter(
                x=self.daily_data['Volume'],
                y=self.daily_data['% Change'],
                mode='markers',
                name="Volume vs Change",
                marker_color=self.daily_data['% Change'],
                marker_colorscale='RdYlGn'
            ), row=2, col=1)
        
            # Country Distribution (Bar)
            country_dist = self.daily_data['Country'].value_counts().head(10)
        
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
        except Exception as e:
            print(f"Error creating market overview dashboard: {str(e)}")
            return go.Figure()
