import pandas as pd

def loss_triangle_builder(df: pd.DataFrame) -> pd.DataFrame:
    req_cols = ["accident_year", "development_year", "paid_loss"]

    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"DataFrame must contain the following columns: {req_cols}")

    df = df.copy()

    if "calendar_year" not in df.columns:
        df["calendar_year"] = (
            df["accident_year"] + (df["development_year"] // 12) - 1
        )

    valuation_year = df["calendar_year"].max()

    df = df[df["calendar_year"] <= valuation_year]

    triangle = df.pivot_table(
        index="accident_year",
        columns="development_year",
        values="paid_loss",
        aggfunc="sum"
    )

    triangle = triangle.sort_index().sort_index(axis=1)

    return triangle