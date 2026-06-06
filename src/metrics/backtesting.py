import pandas as pd
import numpy as np 

from src.models.chain_ladder import chain_ladder
from src.data.triangle_builder import loss_triangle_builder

"""
This function constructs a new loss triangle for backtesting by truncating the original triangle at a specified stop year.
"""
def backtest_loss_triangle (df: pd.DataFrame, stop_year: int) -> pd.DataFrame:
    new_triangle = df.copy()
    for year in new_triangle.columns:
        if (year > stop_year):
            new_triangle[year] = np.nan
    return new_triangle

def cal_errors (df: pd.DataFrame) -> dict[str, float]:
    actual = df['actual_ultimate_loss']
    pred = df['predicted_ultimate_loss']

    errors = pred - actual
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    mape = np.mean(np.abs(errors / actual)) * 100
    bias = np.mean(errors)

    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "Bias": bias
    }

def backtest (original_triangle: pd.DataFrame, stop_year -> int) -> tuple[pd.DataFrame, dict]:
    
    