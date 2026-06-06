import pandas as pd
import numpy as np

from src.models.chain_ladder import chain_ladder


def backtest_loss_triangle(
    df: pd.DataFrame,
    stop_calendar_year: int
) -> pd.DataFrame:
    new_triangle = df.copy()

    for accident_year in new_triangle.index:
        for development_period in new_triangle.columns:
            development_age_years = int(development_period) // 12
            cell_calendar_year = int(accident_year) + development_age_years - 1

            if cell_calendar_year > stop_calendar_year:
                new_triangle.loc[accident_year, development_period] = np.nan

    return new_triangle


def cal_errors(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "Bias": np.nan
        }

    actual = df["actual_ultimate_loss"]
    pred = df["predicted_ultimate_loss"]

    errors = pred - actual

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    nonzero_actual = actual != 0
    if nonzero_actual.any():
        mape = np.mean(np.abs(errors[nonzero_actual] / actual[nonzero_actual])) * 100
    else:
        mape = np.nan

    bias = np.mean(errors)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias
    }


def backtest(
    original_triangle: pd.DataFrame,
    stop_calendar_year: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    backtest_triangle = backtest_loss_triangle(
        original_triangle,
        stop_calendar_year
    )

    # Drop future accident years with no observed data as of the stop year
    backtest_triangle = backtest_triangle.dropna(axis=0, how="all")

    # Drop future development periods with no observed data as of the stop year
    backtest_triangle = backtest_triangle.dropna(axis=1, how="all")

    _, _, pred = chain_ladder(backtest_triangle)

    actual = original_triangle.apply(
        lambda row: row.dropna().iloc[-1],
        axis=1
    )

    results = []

    for _, row in pred.iterrows():
        accident_year = row["accident_year"]

        if accident_year not in actual.index:
            continue

        predicted_ultimate_loss = row["projected_ultimate_loss"]
        actual_ultimate_loss = actual.loc[accident_year]

        results.append({
            "accident_year": accident_year,
            "predicted_ultimate_loss": predicted_ultimate_loss,
            "actual_ultimate_loss": actual_ultimate_loss
        })

    results_df = pd.DataFrame(
        results,
        columns=[
            "accident_year",
            "predicted_ultimate_loss",
            "actual_ultimate_loss"
        ]
    )

    return results_df, cal_errors(results_df)