import pandas as pd
import numpy as np 

from src.models.chain_ladder import chain_ladder

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
    if df.empty:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "Bias": np.nan
        }

    actual = df['actual_ultimate_loss']
    pred = df['predicted_ultimate_loss']

    errors = pred - actual
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / actual)) * 100
    bias = np.mean(errors)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias
    }

def backtest (original_triangle: pd.DataFrame, stop_year: int) -> tuple[pd.DataFrame, dict]:
    new_triangle = backtest_loss_triangle(original_triangle, stop_year)
    new_triangle = new_triangle.dropna(axis=1, how="all")
    _,_,pred= chain_ladder(new_triangle)
    actual = original_triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)

    results = list()
    for _, row in pred.iterrows():
        accident_year = row['accident_year']
        predicted_ultimate_loss = row['projected_ultimate_loss']
        actual_ultimate_loss = actual.loc[accident_year]
        results.append({
            "accident_year": accident_year,
            "predicted_ultimate_loss": predicted_ultimate_loss,
            "actual_ultimate_loss": actual_ultimate_loss
        })

    results_df = pd.DataFrame(
        results,
        columns=["accident_year", "predicted_ultimate_loss", "actual_ultimate_loss"]
    )
    return results_df, cal_errors(results_df)