  1 """
  2 HTML Report Generator for Financial Analysis
  3 Creates comprehensive downloadable HTML reports with interactive charts
  4 """
  5 
  6 import pandas as pd
  7 import plotly.graph_objects as go
  8 from plotly.subplots import make_subplots
  9 from datetime import datetime
 10 from typing import Dict, List, Optional
 11 import base64
 12 from io import StringIO
 13 
 14 class HTMLReportGenerator:
 15     """Generate comprehensive HTML reports with interactive charts and analysis."""
 16     
 17     def _clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
 18         """Clean DataFrame to resolve index/column ambiguity issues."""
 19         if data is None or data.empty:
 20             return data
 21 
 22         # Create a copy to avoid modifying original data
 23         data = data.copy()
 24 
 25         try:
 26             # Reset index to avoid any datetime index conflicts
 27             if isinstance(data.index, pd.DatetimeIndex) or data.index.name == 'Datetime':
 28                 data = data.reset_index()
 29 
 30             # Remove any 'Datetime' or 'Date' columns to prevent ambiguity
 31             for col in ['Datetime', 'Date']:
 32                 if col in data.columns:
 33                     data = data.drop(columns=[col], errors='ignore')
 34 
 35             # Ensure required columns exist
 36             required_cols = ['Close']
 37             if not all(col in data.columns for col in required_cols):
 38                 raise ValueError("Missing required columns: {}".format([col for col in required_cols if col not in data.columns]))
 39 
 40             # Set a datetime index if none exists
 41             if not isinstance(data.index, pd.DatetimeIndex):
 42                 # Create a synthetic datetime index based on data length
 43                 data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
 44                 data.index.name = 'Datetime'
 45 
 46         except Exception as e:
 47             # Log error and return a simplified version
 48             import logging
 49             logging.basicConfig(level=logging.INFO)
 50             logger = logging.getLogger(__name__)
 51             logger.error(f"Error cleaning DataFrame: {str(e)}")
 52             try:
 53                 data = data.reset_index(drop=True)
 54                 for col in ['Datetime', 'Date']:
 55                     if col in data.columns:
 56                         data = data.drop(columns=[col], errors='ignore')
 57             except:
 58                 pass  # Return original data if all fixes fail
 59 
 60         return data
 61 
 62     def __init__(self):
 63         self.css_styles = """
 64         
 65         """
 66     
 67     def generate_comprehensive_report(self, 
 68                                     stock_symbol: str,
 69                                     historical_data: pd.DataFrame,
 70                                     tech_indicators,
 71                                     analytics,
 72                                     visualizations,
 73                                     prediction_data=None,
 74                                     prediction_days=None,
 75                                     advanced_analytics=None,
 76                                     report_type: str = "full") -> str:
 77         """Generate a comprehensive HTML report with all analysis components."""
 78         
 79         # Clean data to resolve ambiguity issues
 80         historical_data = self._clean_dataframe(historical_data)
 81         
 82         # Generate timestamp
 83         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 84         
 85         # Start building HTML
 86         html_content = f"""
 87         
 88         
 89             {self.css_styles}
 90         
 91         
 92         """
 93         
 94         # Header
 95         html_content += f"""
 96         
 97             Financial Analysis Report
 98 
 99             Stock Symbol: {stock_symbol} | Generated: {timestamp}
100 
101         
102 
103         """
104         
105         # Executive Summary
106         html_content += self._generate_executive_summary(stock_symbol, historical_data, tech_indicators)
107         
108         # Technical Indicators Charts
109         html_content += self._generate_technical_charts_section(tech_indicators)
110         
111         # Trading Signals
112         html_content += self._generate_trading_signals_section(tech_indicators)
113         
114         # Price Visualization Charts
115         html_content += self._generate_price_charts_section(visualizations)
116         
117         # Performance Metrics
118         html_content += self._generate_performance_metrics(historical_data)
119         
120         # Risk Analysis
121         html_content += self._generate_risk_analysis(analytics, historical_data)
122         
123         # Add prediction charts if available
124         if prediction_data is not None and report_type in ["full", "predictions"]:
125             html_content += self._generate_prediction_charts_section(prediction_data, prediction_days, historical_data)
126         
127         # Add 3D visualization charts if available
128         if visualizations is not None and report_type in ["full", "advanced"]:
129             html_content += self._generate_3d_charts_section(visualizations)
130         
131         # Add advanced analytics if available
132         if advanced_analytics is not None and report_type in ["full", "advanced"]:
133             html_content += self._generate_advanced_analytics_section(advanced_analytics)
134         
135         # Footer
136         html_content += f"""
137         
138             This report was generated automatically by the Financial Analysis Application
139 
140             Data analysis period: {str(historical_data.index[0])[:10]} to {str(historical_data.index[-1])[:10]}
141 
142         
143 
144         
145         """
146         
147         return html_content
148     
149     def _generate_executive_summary(self, symbol: str, data: pd.DataFrame, tech_indicators) -> str:
150         """Generate executive summary section."""
151         current_price = data['Close'].iloc[-1]
152         price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
153         price_change_pct = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 and data['Close'].iloc[-2] != 0 else 0
154         
155         # Calculate key metrics
156         high_52week = data['High'].max()
157         low_52week = data['Low'].min()
158         avg_volume = data['Volume'].mean()
159         
160         return f"""
161         
162             📊 Executive Summary
163 
164             
165                 
166                     ${current_price:.2f}
167 
168                     Current Price
169 
170                 
171 
172                 
173                     
174                         {price_change:+.2f} ({price_change_pct:+.2f}%)
175                     
176 
177                     Daily Change
178 
179                 
180 
181                 
182                     ${high_52week:.2f}
183 
184                     52-Week High
185 
186                 
187 
188                 
189                     ${low_52week:.2f}
190 
191                     52-Week Low
192 
193                 
194 
195                 
196                     {avg_volume:,.0f}
197 
198                     Avg Volume
199 
200                 
201 
202             
203 
204         
205 
206         """
207     
208     def _generate_technical_charts_section(self, tech_indicators) -> str:
209         """Generate technical analysis charts section."""
210         html_section = """
211         
212             📈 Technical Analysis Charts
213 
214             Interactive charts showing technical indicators with MM-DD-YYYY date formatting on hover.
215 
216         """
217         
218         try:
219             # Moving Averages Chart
220             ma_chart = tech_indicators.create_moving_averages_chart()
221             html_section += f"""
222             
223                 Moving Averages
224 
225                 {ma_chart.to_html(include_plotlyjs='inline', div_id='ma-chart')}
226             
227 
228             """
229             
230             # RSI Chart
231             rsi_chart = tech_indicators.create_rsi_chart()
232             html_section += f"""
233             
234                 Relative Strength Index (RSI)
235 
236                 {rsi_chart.to_html(include_plotlyjs=False, div_id='rsi-chart')}
237             
238 
239             """
240             
241             # MACD Chart
242             macd_chart = tech_indicators.create_macd_chart()
243             html_section += f"""
244             
245                 MACD Analysis
246 
247                 {macd_chart.to_html(include_plotlyjs=False, div_id='macd-chart')}
248             
249 
250             """
251             
252             # Bollinger Bands Chart
253             bb_chart = tech_indicators.create_bollinger_bands_chart()
254             html_section += f"""
255             
256                 Bollinger Bands
257 
258                 {bb_chart.to_html(include_plotlyjs=False, div_id='bb-chart')}
259             
260 
261             """
262             
263         except Exception as e:
264             html_section += f"Error generating technical charts: {str(e)}<br>"
265         
266         html_section += "<br>"
267         return html_section
268     
269     def _generate_trading_signals_section(self, tech_indicators) -> str:
270         """Generate trading signals analysis section."""
271         html_section = """
272         
273             🎯 Trading Signals & Recommendations
274 
275         """
276         
277         try:
278             signals = tech_indicators.get_trading_signals()
279             
280             for indicator, signal_data in signals.items():
281                 signal_type = signal_data.get('signal', 'Unknown').lower()
282                 strength = signal_data.get('strength', 'Unknown')
283                 
284                 signal_class = 'signal-hold'
285                 if 'buy' in signal_type:
286                     signal_class = 'signal-buy'
287                 elif 'sell' in signal_type:
288                     signal_class = 'signal-sell'
289                 
290                 html_section += f"""
291                 
292                     {indicator}
293 
294                     Signal: {signal_data.get('signal', 'Unknown')}
295 
296                     Strength: {strength}
297 
298                 
299 
300                 """
301                 
302         except Exception as e:
303             html_section += f"Error generating trading signals: {str(e)}<br>"
304         
305         html_section += "<br>"
306         return html_section
307     
308     def _generate_price_charts_section(self, visualizations) -> str:
309         """Generate price visualization charts section."""
310         html_section = """
311         
312             💹 Price Analysis Charts
313 
314         """
315         
316         try:
317             # Candlestick Chart
318             candlestick_chart = visualizations.create_candlestick_chart()
319             html_section += f"""
320             
321                 Candlestick Chart
322 
323                 {candlestick_chart.to_html(include_plotlyjs=False, div_id='candlestick-chart')}
324             
325 
326             """
327             
328             # Price Trends Chart
329             trends_chart = visualizations.create_price_trends_chart()
330             html_section += f"""
331             
332                 Price Trends
333 
334                 {trends_chart.to_html(include_plotlyjs=False, div_id='trends-chart')}
335             
336 
337             """
338             
339             # Volume Analysis Chart
340             volume_chart = visualizations.create_volume_chart()
341             html_section += f"""
342             
343                 Volume Analysis
344 
345                 {volume_chart.to_html(include_plotlyjs=False, div_id='volume-chart')}
346             
347 
348             """
349             
350         except Exception as e:
351             html_section += f"Error generating price charts: {str(e)}<br>"
352         
353         html_section += "<br>"
354         return html_section
355     
356     def _generate_performance_metrics(self, data: pd.DataFrame) -> str:
357         """Generate performance metrics table."""
358         html_section = """
359         
360             📊 Performance Metrics
361 
362             	Metric	Value	Description
363 
364 
365         """
366         
367         try:
368             # Calculate metrics
369             total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
370             volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized
371             max_price = data['High'].max()
372             min_price = data['Low'].min()
373             avg_volume = data['Volume'].mean()
374             
375             metrics = [
376                 ("Total Return", f"{total_return:.2f}%", "Overall price appreciation/depreciation"),
377                 ("Annualized Volatility", f"{volatility:.2f}%", "Price volatility over the period"),
378                 ("Maximum Price", f"${max_price:.2f}", "Highest price reached"),
379                 ("Minimum Price", f"${min_price:.2f}", "Lowest price reached"),
380                 ("Average Volume", f"{avg_volume:,.0f}", "Average daily trading volume"),
381                 ("Data Points", f"{len(data)}", "Number of trading days analyzed")
382             ]
383             
384             for metric, value, description in metrics:
385                 html_section += f"""
386                 	{metric}	{value}	{description}
387 
388 
389                 """
390                 
391         except Exception as e:
392             html_section += f"	Error calculating metrics: {str(e)}
393 
394 """
395         
396         html_section += """
397                 
398             
399         
400 
401         """
402         return html_section
403     
404     def _generate_risk_analysis(self, analytics, data: pd.DataFrame) -> str:
405         """Generate risk analysis section."""
406         html_section = """
407         
408             ⚠️ Risk Analysis
409 
410         """
411         
412         try:
413             # Calculate risk metrics
414             returns = data['Close'].pct_change().dropna()
415             volatility = returns.std() * (252 ** 0.5) * 100
416             max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
417             
418             html_section += f"""
419             
420                 
421                     {volatility:.2f}%
422 
423                     Annualized Volatility
424 
425                 
426 
427                 
428                     {max_drawdown:.2f}%
429 
430                     Maximum Drawdown
431 
432                 
433 
434             
435 
436             Risk Assessment
437 
438             
439             """
440             
441             if volatility > 30:
442                 html_section += "🔴 High Risk: This stock shows high volatility. Suitable for experienced traders with high risk tolerance."
443             elif volatility > 20:
444                 html_section += "🟡 Medium Risk: Moderate volatility. Suitable for balanced investment strategies."
445             else:
446                 html_section += "🟢 Low Risk: Relatively stable price movements. Suitable for conservative investors."
447             
448             html_section += "<br>"
449             
450         except Exception as e:
451             html_section += f"Error calculating risk metrics: {str(e)}<br>"
452         
453         html_section += "<br>"
454         return html_section
455     
456     def save_report_to_file(self, html_content: str, filename: Optional[str] = None) -> str:
457         """Save HTML report to file and return the filename."""
458         if filename is None:
459             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
460             filename = f"financial_analysis_report_{timestamp}.html"
461         
462         with open(filename, 'w', encoding='utf-8') as f:
463             f.write(html_content)
464         
465         return filename
466     
467     def _generate_prediction_charts_section(self, prediction_data, prediction_days, historical_data) -> str:
468         """Generate prediction charts section for HTML report using stored prediction data."""
469         html_section = """
470         
471             📈 Price Predictions
472 
473         """
474         
475         if prediction_data is None:
476             html_section += """
477             No price predictions generated. Generate predictions in the application first to include them in the report.
478 
479             
480 
481             """
482             return html_section
483         
484         html_section += f"Price predictions for {prediction_days} days using {prediction_data.get('method', 'selected')} method.<br>"
485         
486         try:
487             # Get stored prediction results
488             pred_prices = prediction_data.get('prices', [])
489             method = prediction_data.get('method', 'technical_analysis')
490             confidence_data = prediction_data.get('confidence', {})
491             disclaimer = prediction_data.get('disclaimer', '')
492             
493             if pred_prices and len(pred_prices) > 0:
494                 # Create prediction chart using stored data
495                 import plotly.graph_objects as go
496                 import pandas as pd
497                 from datetime import datetime, timedelta
498                 
499                 # Get recent historical prices for context
500                 recent_data = historical_data.tail(20)
501                 historical_dates = recent_data.index
502                 historical_prices = recent_data['Close'].values
503                 
504                 # Generate future dates for predictions
505                 if isinstance(historical_dates, pd.DatetimeIndex) and len(historical_dates) > 0:
506                     last_date = historical_dates[-1]
507                 else:
508                     last_date = pd.Timestamp.now()
509                 
510                 future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(pred_prices), freq='D')
511                 
512                 # Create chart
513                 fig = go.Figure()
514                 
515                 # Historical prices
516                 fig.add_trace(go.Scatter(
517                     x=historical_dates,
518                     y=historical_prices,
519                     mode='lines',
520                     name='Historical Prices',
521                     line=dict(color='blue', width=2),
522                     hovertemplate='Date: %{x|%m-%d-%Y}<br>Price: $%{y:.2f}'
523                 ))
524                 
525                 # Predicted prices
526                 method_name = {
527                     'technical_analysis': 'Technical Analysis',
528                     'linear_trend': 'Linear Trend',
529                     'moving_average': 'Moving Average'
530                 }.get(method, method.replace('_', ' ').title())
531                 
532                 fig.add_trace(go.Scatter(
533                     x=future_dates,
534                     y=pred_prices,
535                     mode='lines+markers',
536                     name=f'{method_name} Prediction',
537                     line=dict(color='red', width=2, dash='dash'),
538                     marker=dict(size=6),
539                     hovertemplate='Date: %{x|%m-%d-%Y}<br>Predicted: $%{y:.2f}'
540                 ))
541                 
542                 # Update layout with MM-DD-YYYY format
543                 fig.update_layout(
544                     title=f'{method_name} Price Prediction - {prediction_days} Days',
545                     xaxis_title='Date',
546                     yaxis_title='Price ($)',
547                     hovermode='x unified',
548                     showlegend=True,
549                     height=400,
550                     xaxis=dict(
551                         type='date',
552                         showticklabels=True,
553                         showgrid=True,
554                         hoverformat='%m-%d-%Y'
555                     )
556                 )
557                 
558                 chart_html = fig.to_html(include_plotlyjs=False, div_id=f"prediction_chart")
559                 
560                 html_section += f"""
561                 
562                     {method_name} Prediction Chart
563 
564                     {chart_html}
565                 
566 
567                 """
568                 
569                 # Add prediction table for easy readability
570                 current_date = datetime.now()
571                 pred_dates = [current_date + timedelta(days=i+1) for i in range(len(pred_prices))]
572                 
573                 html_section += f"""
574                 
575                     Predicted Prices Table
576 
577                     
578                         	Date	Day	Predicted Price
579 
580 
581                 """
582                 
583                 for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
584                     html_section += f"""
585                             	{date.strftime('%Y-%m-%d')}	Day {i+1}	${price:.2f}
586 
587 
588                     """
589                 
590                 html_section += """
591                         
592                     
593 
594                 
595 
596                 """
597                 
598                 # Add prediction metrics and disclaimer
599                 if confidence_data:
600                     # Fix confidence level formatting - handle string values
601                     confidence_level = confidence_data.get('confidence_level', 'N/A')
602                     if isinstance(confidence_level, str) and '%' in confidence_level:
603                         confidence_display = confidence_level
604                     else:
605                         try:
606                             confidence_display = f"{float(confidence_level):.1f}%"
607                         except:
608                             confidence_display = str(confidence_level)
609                     
610                     volatility_risk = confidence_data.get('volatility_risk', 0)
611                     try:
612                         volatility_display = f"{float(volatility_risk):.2f}%"
613                     except:
614                         volatility_display = str(volatility_risk)
615                     
616                     html_section += f"""
617                     
618                         Prediction Metrics & Reliability
619 
620                         
621                             	Metric	Value
622 	Confidence Level	{confidence_display}
623 	Trend Strength	{confidence_data.get('trend_strength', 'N/A')}
624 	Data Quality	{confidence_data.get('data_quality', 'N/A')}
625 	Volatility Risk	{volatility_display}
626 
627 
628                         
629 
630                             {disclaimer}
631 
632                         
633 
634                     
635 
636                     """
637                 else:
638                     html_section += "No prediction data available.<br>"
639                 
640         except Exception as e:
641             html_section += f"Error generating prediction charts: {str(e)}<br>"
642         
643         html_section += ""
644         return html_section
645     
646     def _generate_3d_charts_section(self, visualizations) -> str:
647         """Generate 3D visualization charts section for HTML report."""
648         html_section = """
649         
650             📊 3D Visualizations
651 
652             Interactive three-dimensional analysis for comprehensive market insights.
653 
654         """
655         
656         try:
657             # Check if visualizations has 3D chart methods
658             if hasattr(visualizations, 'get_3d_price_volume_chart'):
659                 chart = visualizations.get_3d_price_volume_chart()
660                 if chart:
661                     chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_price_volume")
662                     html_section += f"""
663                     
664                         3D Price-Volume Analysis
665 
666                         Three-dimensional visualization of price movements, volume, and time relationships.
667 
668                         {chart_html}
669                     
670 
671                     """
672             
673             if hasattr(visualizations, 'get_3d_technical_surface'):
674                 chart = visualizations.get_3d_technical_surface()
675                 if chart:
676                     chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_technical_surface")
677                     html_section += f"""
678                     
679                         3D Technical Indicator Surface
680 
681                         Surface plot showing relationships between multiple technical indicators.
682 
683                         {chart_html}
684                     
685 
686                     """
687             
688             if hasattr(visualizations, 'get_3d_market_dynamics'):
689                 chart = visualizations.get_3d_market_dynamics()
690                 if chart:
691                     chart_html = chart.to_html(include_plotlyjs=False, div_id="3d_market_dynamics")
692                     html_section += f"""
693                     
694                         3D Market Dynamics
695 
696                         Multi-dimensional view of market behavior and trading patterns.
697 
698                         {chart_html}
699                     
700 
701                     """
702             
703         except Exception as e:
704             html_section += f"Error generating 3D charts: {str(e)}<br>"
705         
706         html_section += "<br>"
707         return html_section
708     
709     def _generate_advanced_analytics_section(self, advanced_analytics) -> str:
710         """Generate advanced analytics section for HTML report."""
711         html_section = """
712         
713             🎯 Advanced Analytics
714 
715             Comprehensive sector analysis, correlations, and market intelligence.
716 
717         """
718         
719         try:
720             # Sector Performance Analysis
721             if hasattr(advanced_analytics, 'create_sector_performance_chart'):
722                 chart = advanced_analytics.create_sector_performance_chart()
723                 if chart:
724                     chart_html = chart.to_html(include_plotlyjs=False, div_id="sector_performance")
725                     html_section += f"""
726                     
727                         Sector Performance Analysis
728 
729                         Comparative performance across different market sectors.
730 
731                         {chart_html}
732                     
733 
734                     """
735             
736             # Correlation Heatmap
737             if hasattr(advanced_analytics, 'create_correlation_heatmap'):
738                 chart = advanced_analytics.create_correlation_heatmap()
739                 if chart:
740                     chart_html = chart.to_html(include_plotlyjs=False, div_id="correlation_heatmap")
741                     html_section += f"""
742                     
743                         Market Correlation Analysis
744 
745                         Heat map showing correlations between different market metrics and indicators.
746 
747                         {chart_html}
748                     
749 
750                     """
751             
752             # Performance Dashboard
753             if hasattr(advanced_analytics, 'create_performance_dashboard'):
754                 chart = advanced_analytics.create_performance_dashboard()
755                 if chart:
756                     chart_html = chart.to_html(include_plotlyjs=False, div_id="performance_dashboard")
757                     html_section += f"""
758                     
759                         Comprehensive Performance Dashboard
760 
761                         Multi-metric dashboard showing key performance indicators and trends.
762 
763                         {chart_html}
764                     
765 
766                     """
767             
768             # Industry Analysis
769             if hasattr(advanced_analytics, 'get_industry_analysis'):
770                 industry_data = advanced_analytics.get_industry_analysis()
771                 if industry_data is not None and not industry_data.empty:
772                     html_section += """
773                     
774                         Industry Analysis Summary
775 
776                         
777                     """
778                     html_section += industry_data.head(10).to_html(classes="summary-table", escape=False)
779                     html_section += "<br>"
780             
781         except Exception as e:
782             html_section += f"Error generating advanced analytics: {str(e)}<br>"
783         
784         html_section += "<br>"
785         return html_section
