"""
Interactive Prediction Charts Module

Creates interactive visualizations showing GMADL predictions, strategy thresholds,
and trading signals using Plotly for 3-year Bitcoin trading data.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os
from datetime import datetime


class InteractivePredictionVisualizer:
    """Creates interactive prediction and threshold visualizations."""
    
    def __init__(self, output_dir: str = None, figsize: tuple = (1400, 800)):
        """
        Initialize interactive visualizer.
        
        Args:
            output_dir: Directory for saving HTML files
            figsize: Figure size (width, height) in pixels
        """
        self.output_dir = output_dir
        self.figsize = figsize
        
        # Create output directory if provided
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Professional color scheme
        self.colors = {
            'prediction': '#2E86AB',
            'price': '#1f77b4',
            'enter_long': '#4CAF50',    # Green
            'exit_long': '#FFC107',     # Amber  
            'enter_short': '#F44336',   # Red
            'exit_short': '#FF9800',    # Orange
            'long_signal': '#4CAF50',
            'short_signal': '#F44336',
            'exit_signal': '#9E9E9E',
            'background_long': 'rgba(76, 175, 80, 0.1)',
            'background_short': 'rgba(244, 67, 54, 0.1)',
            'portfolio': '#9C27B0'
        }
    
    def extract_prediction_data(self, strategy, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract prediction data and thresholds from GMADL strategy.
        
        Args:
            strategy: ModelGmadlPredictionsStrategy instance
            data: Market data DataFrame
            
        Returns:
            Tuple of (prediction_data_df, thresholds_dict)
        """
        # Get strategy info for thresholds
        strategy_info = strategy.info()
        thresholds = {
            'enter_long': strategy_info.get('enter_long'),
            'exit_long': strategy_info.get('exit_long'), 
            'enter_short': strategy_info.get('enter_short'),
            'exit_short': strategy_info.get('exit_short')
        }
        
        # Merge predictions with data to get timestamps
        merged_data = pd.merge(
            data, strategy.predictions, on=['time_index', 'group_id'], how='left'
        )
        
        # Extract prediction values (assuming future=1, feature=0)
        prediction_values = []
        timestamps = []
        
        for idx, row in merged_data.iterrows():
            prediction_val = row['prediction']
            
            # Check if prediction is not NaN and not None with proper array handling
            is_valid_prediction = False
            try:
                if prediction_val is not None:
                    if isinstance(prediction_val, np.ndarray):
                        is_valid_prediction = prediction_val.size > 0
                    elif not pd.isna(prediction_val):
                        is_valid_prediction = True
                        
            except (TypeError, ValueError):
                is_valid_prediction = False
            
            if is_valid_prediction:
                try:
                    # Extract prediction array
                    pred_array = prediction_val
                    if isinstance(pred_array, np.ndarray) and pred_array.size > 0:
                        # Get prediction for future=1, feature=0
                        if len(pred_array.shape) == 3:
                            # Shape: (batch, future_steps, features)
                            future_idx = min(strategy.future, pred_array.shape[1] - 1)
                            pred_value = pred_array[0, future_idx, 0]
                        elif len(pred_array.shape) == 2:
                            # Shape: (future_steps, features) 
                            future_idx = min(strategy.future, pred_array.shape[0] - 1)
                            pred_value = pred_array[future_idx, 0] if pred_array.shape[1] > 0 else pred_array[future_idx]
                        elif len(pred_array.shape) == 1:
                            # Shape: (features,)
                            pred_value = pred_array[0]
                        else:
                            # Scalar
                            pred_value = float(pred_array)
                        
                        prediction_values.append(pred_value)
                    else:
                        prediction_values.append(np.nan)
                except Exception as e:
                    print(f"Warning: Could not extract prediction value: {e}")
                    prediction_values.append(np.nan)
            else:
                prediction_values.append(np.nan)
            
            # Handle timestamp - use whichever is available
            if 'timestamp' in row:
                timestamps.append(row['timestamp'])
            elif 'close_time' in row:
                timestamps.append(row['close_time'])
            else:
                timestamps.append(None)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': prediction_values,
            'close_price': merged_data['close_price'].values
        })
        
        # Remove NaN predictions and invalid timestamps for cleaner visualization
        pred_df = pred_df.dropna(subset=['prediction', 'timestamp'])
        
        return pred_df, thresholds
    
    def generate_trading_signals(self, pred_df: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """
        Generate trading signals based on prediction threshold crossings.
        
        Args:
            pred_df: DataFrame with predictions and timestamps
            thresholds: Dictionary of threshold values
            
        Returns:
            DataFrame with signal information
        """
        signals = []
        current_position = 0  # 0=exit, 1=long, -1=short
        
        for idx, row in pred_df.iterrows():
            pred_value = row['prediction']
            timestamp = row['timestamp']
            price = row['close_price']
            
            signal_type = None
            signal_color = None
            
            # Check for signal triggers with proper None handling
            try:
                if (thresholds.get('enter_long') is not None and 
                    not np.isnan(pred_value) and
                    pred_value > thresholds['enter_long'] and current_position != 1):
                    signal_type = 'Enter Long'
                    signal_color = self.colors['long_signal']
                    current_position = 1
                    
                elif (thresholds.get('enter_short') is not None and 
                      not np.isnan(pred_value) and
                      pred_value < thresholds['enter_short'] and current_position != -1):
                    signal_type = 'Enter Short'
                    signal_color = self.colors['short_signal']
                    current_position = -1
                    
                elif (thresholds.get('exit_long') is not None and 
                      not np.isnan(pred_value) and
                      pred_value < thresholds['exit_long'] and current_position == 1):
                    signal_type = 'Exit Long'
                    signal_color = self.colors['exit_signal']
                    current_position = 0
                    
                elif (thresholds.get('exit_short') is not None and 
                      not np.isnan(pred_value) and
                      pred_value > thresholds['exit_short'] and current_position == -1):
                    signal_type = 'Exit Short'
                    signal_color = self.colors['exit_signal']
                    current_position = 0
            except Exception as e:
                print(f"Warning: Signal generation error for value {pred_value}: {e}")
                continue
            
            if signal_type:
                signals.append({
                    'timestamp': timestamp,
                    'prediction': pred_value,
                    'price': price,
                    'signal': signal_type,
                    'color': signal_color,
                    'position': current_position
                })
        
        return pd.DataFrame(signals)
    
    def create_interactive_prediction_plot(self, strategy, data: pd.DataFrame,
                                         title: str = "GMADL Predictions & Trading Signals",
                                         save_path: str = None,
                                         portfolio_values: np.ndarray = None) -> str:
        """
        Create comprehensive interactive prediction visualization.
        
        Args:
            strategy: ModelGmadlPredictionsStrategy instance
            data: Market data DataFrame
            title: Chart title
            save_path: Path to save HTML file
            portfolio_values: Optional portfolio values for subplot
            
        Returns:
            Path to saved HTML file
        """
        # Extract prediction data
        try:
            pred_df, thresholds = self.extract_prediction_data(strategy, data)
        except Exception as e:
            print(f"Error extracting prediction data: {e}")
            return None
        
        if pred_df.empty:
            print("No prediction data available for visualization")
            return None
        
        # Generate trading signals
        signals_df = self.generate_trading_signals(pred_df, thresholds)
        
        # Create subplot structure
        subplot_titles = ["GMADL Predictions & Thresholds", "Bitcoin Price"]
        if portfolio_values is not None:
            subplot_titles.append("Portfolio Value")
            rows = 3
            row_heights = [0.5, 0.25, 0.25]
        else:
            rows = 2  
            row_heights = [0.7, 0.3]
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Main prediction plot
        fig.add_trace(
            go.Scatter(
                x=pred_df['timestamp'],
                y=pred_df['prediction'],
                mode='lines',
                name='GMADL Prediction',
                line=dict(color=self.colors['prediction'], width=2),
                hovertemplate='<b>Prediction</b><br>' +
                            'Time: %{x}<br>' +
                            'Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add threshold lines
        x_range = [pred_df['timestamp'].min(), pred_df['timestamp'].max()]
        
        for threshold_name, threshold_value in thresholds.items():
            if threshold_value is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[threshold_value, threshold_value],
                        mode='lines',
                        name=f'{threshold_name.replace("_", " ").title()}: {threshold_value:.3f}',
                        line=dict(
                            color=self.colors[threshold_name],
                            width=2,
                            dash='dash'
                        ),
                        showlegend=True,
                        hovertemplate=f'<b>{threshold_name.replace("_", " ").title()}</b><br>' +
                                    f'Threshold: {threshold_value:.4f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add trading signals
        if not signals_df.empty:
            for signal_type in signals_df['signal'].unique():
                signal_data = signals_df[signals_df['signal'] == signal_type]
                
                fig.add_trace(
                    go.Scatter(
                        x=signal_data['timestamp'],
                        y=signal_data['prediction'],
                        mode='markers',
                        name=f'{signal_type} Signal',
                        marker=dict(
                            color=signal_data['color'].iloc[0],
                            size=10,
                            symbol='diamond',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=f'<b>{signal_type}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Prediction: %{y:.4f}<br>' +
                                    'Price: %{customdata:.2f}<extra></extra>',
                        customdata=signal_data['price']
                    ),
                    row=1, col=1
                )
        
        # Bitcoin price subplot
        fig.add_trace(
            go.Scatter(
                x=pred_df['timestamp'],
                y=pred_df['close_price'],
                mode='lines',
                name='Bitcoin Price',
                line=dict(color=self.colors['price'], width=1),
                hovertemplate='<b>Bitcoin Price</b><br>' +
                            'Time: %{x}<br>' +
                            'Price: $%{y:,.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Portfolio value subplot (if provided)
        if portfolio_values is not None and len(portfolio_values) > 0:
            # Align portfolio values with timestamps (assuming same length)
            portfolio_timestamps = data['timestamp'].iloc[:len(portfolio_values)]
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_timestamps,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.colors['portfolio'], width=2),
                    fill='tonexty',
                    hovertemplate='<b>Portfolio Value</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Update layout for interactivity
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            width=self.figsize[0],
            height=self.figsize[1],
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                type='date',
                title='Time'
            ),
            template='plotly_white'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Prediction Value", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        if portfolio_values is not None:
            fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
        
        # Add range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        if save_path:
            fig.write_html(save_path, include_plotlyjs=True)
            print(f"Interactive prediction chart saved to: {save_path}")
            return save_path
        else:
            fig.show()
            return None
    
    def create_prediction_summary_stats(self, pred_df: pd.DataFrame, 
                                       signals_df: pd.DataFrame,
                                       thresholds: Dict) -> Dict[str, Any]:
        """
        Create summary statistics for the prediction visualization.
        
        Args:
            pred_df: Prediction data DataFrame
            signals_df: Trading signals DataFrame  
            thresholds: Threshold values
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_predictions': len(pred_df),
            'prediction_period': {
                'start': pred_df['timestamp'].min(),
                'end': pred_df['timestamp'].max(),
                'duration_days': (pred_df['timestamp'].max() - pred_df['timestamp'].min()).days
            },
            'prediction_stats': {
                'mean': pred_df['prediction'].mean(),
                'std': pred_df['prediction'].std(),
                'min': pred_df['prediction'].min(),
                'max': pred_df['prediction'].max(),
                'median': pred_df['prediction'].median()
            },
            'thresholds': thresholds,
            'signals': {
                'total_signals': len(signals_df),
                'signal_breakdown': signals_df['signal'].value_counts().to_dict() if not signals_df.empty else {}
            }
        }
        
        return stats