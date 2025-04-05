"""
Plotting Module
~~~~~~~~~~~~~~~

This module provides an alternate set of plotting functions for visualizing model
predictions and time series data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import cupy as cp

# Set default renderer.
pio.renderers.default = 'colab'


def create_scatter_plot(test_df, pred_lin, pred_xgb, target_col, basin_name):
    """
    Creates a scatter plot comparing actual vs. predicted SWE values.

    :param test_df: DataFrame with test set data.
    :param pred_lin: Predictions from the linear regression model.
    :param pred_xgb: Predictions from the XGBoost model.
    :param target_col: Name of the target column.
    :param basin_name: Name of the basin.
    :return: Plotly Figure.
    """
    actual = test_df[target_col].to_numpy()
    min_val = min(actual.min(), pred_lin.min(), pred_xgb.min())
    max_val = max(actual.max(), pred_lin.max(), pred_xgb.max())

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=actual, y=pred_lin, mode='markers', name='Linear Regression',
        marker=dict(color='cyan', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=actual, y=pred_xgb, mode='markers', name='XGBoost',
        marker=dict(color='yellow', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='1:1 Line',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f"{basin_name}: Actual vs Predicted SWE (Test Set)",
        xaxis_title="Actual (Bias-Corrected) SWE (mm)",
        yaxis_title="Predicted SWE (mm)",
        template="plotly_dark"
    )
    return fig


def create_timeseries_plot(test_df, pred_lin, pred_xgb, target_col, basin_name):
    """
    Creates a time series plot for the test set comparing original, adjusted, and predicted SWE.

    :param test_df: DataFrame with test set data.
    :param pred_lin: Predictions from the linear regression model.
    :param pred_xgb: Predictions from the XGBoost model.
    :param target_col: Name of the target column.
    :param basin_name: Name of the basin.
    :return: Plotly Figure.
    """
    test_df.index = pd.to_datetime(test_df.index)
    x = np.array(test_df.index)
    y_original = test_df['pillow_swe'].to_numpy()
    y_adjusted = test_df[target_col].to_numpy()
    y_lin = np.array(pred_lin)
    y_xgb = np.array(pred_xgb)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_original, mode='lines',
        name='Original SWE',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_adjusted, mode='lines',
        name='Adjusted SWE',
        line=dict(color='cyan', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_lin, mode='lines',
        name='Linear Regression',
        line=dict(color='cyan', dash='dot', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_xgb, mode='lines',
        name='XGBoost',
        line=dict(color='yellow', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=test_df['aso_swe'].to_numpy(), mode='markers',
        name='ASO SWE',
        marker=dict(color='orange', size=8)
    ))
    fig.update_layout(
        title=f"{basin_name}: Time Series (Test Set)",
        xaxis_title="Date",
        yaxis_title="SWE (mm)",
        template="plotly_dark",
        showlegend=True
    )
    return fig


def create_timeseries_plot_train(train_df, pred_lin, pred_xgb, target_col, basin_name):
    """
    Creates a time series plot for the training set showing original, adjusted, and predicted SWE.

    :param train_df: DataFrame with training set data.
    :param pred_lin: Predictions from the linear regression model.
    :param pred_xgb: Predictions from the XGBoost model.
    :param target_col: Name of the target column.
    :param basin_name: Name of the basin.
    :return: Plotly Figure.
    """
    train_df.index = pd.to_datetime(train_df.index)
    x = np.array(train_df.index)
    y_original = train_df['pillow_swe'].to_numpy()
    y_adjusted = train_df[target_col].to_numpy()
    y_lin = np.array(pred_lin)
    y_xgb = np.array(pred_xgb)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_original, mode='lines',
        name='Original SWE',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_adjusted, mode='lines',
        name='Adjusted SWE',
        line=dict(color='cyan', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_lin, mode='lines',
        name='Linear Regression',
        line=dict(color='cyan', dash='dot', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_xgb, mode='lines',
        name='XGBoost',
        line=dict(color='yellow', width=2)
    ))
    if 'aso_swe' in train_df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=train_df['aso_swe'].to_numpy(), mode='markers',
            name='ASO SWE',
            marker=dict(color='orange', size=8)
        ))
    fig.update_layout(
        title=f"{basin_name}: Time Series (Training Set)",
        xaxis_title="Date",
        yaxis_title="SWE (mm)",
        template="plotly_dark",
        showlegend=True
    )
    return fig


def plot_feature_importances(model, features, title):
    """
    Plots the feature importances or absolute coefficients for the given model.

    :param model: A trained model with feature_importances_ or coef_ attribute.
    :param features: List of feature names.
    :param title: Title for the plot.
    :return: Plotly Figure.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        if hasattr(importance, 'get'):
            importance = importance.get()
        importance = np.abs(importance).flatten()
    elif hasattr(model, 'coef_'):
        try:
            importance = cp.asnumpy(model.coef_).flatten()
        except Exception:
            importance = np.array(model.coef_).flatten()
        importance = np.abs(importance)
    else:
        importance = np.zeros(len(features))
        print("Warning: Model has no feature importances or coefficients.")

    fig = go.Figure(go.Bar(x=features, y=importance, marker_color='magenta'))
    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Importance",
        template="plotly_dark"
    )
    return fig
