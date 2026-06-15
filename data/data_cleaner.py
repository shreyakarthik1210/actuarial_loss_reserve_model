import pandas as pd
import re

STANDARD_COLUMNS = {
    "accident_year": [
        "accident_year",
        "accident year",
        "accidentyear",
        "ay",
        "loss_year",
        "loss year",
        "origin_year",
        "origin year",
        "policy_year",
        "policy year"
    ],
    "development_year": [
        "development_year",
        "development year",
        "developmentyear",
        "development_period",
        "development period",
        "dev_year",
        "dev year",
        "dev",
        "development_month",
        "development month",
        "developmentmonth",
        "dev_month",
        "dev month",
        "months_reported",
        "months reported",
        "age",
        "age_months"
    ],
    "paid_loss": [
        "paid_loss",
        "paid loss",
        "paidloss",
        "paid",
        "cumulative_paid",
        "cumulative paid",
        "cumulative_paid_loss",
        "cumulative paid loss",
        "cum_paid",
        "cum paid",
        "loss_paid",
        "loss paid",
        "amount_paid",
        "amount paid",
        "cumpaidloss"
    ],
    "incurred_loss": [
        "incurred_loss",
        "incurred loss",
        "incurredloss",
        "incurred",
        "cumulative_incurred",
        "cumulative incurred",
        "cumulative_incurred_loss",
        "cumulative incurred loss"
    ],
    "claim_count": [
        "claim_count",
        "claim count",
        "claimcount",
        "claims",
        "number_of_claims",
        "number of claims",
        "reported_claims",
        "reported claims"
    ],
    "line_of_business": [
        "line_of_business",
        "line of business",
        "lob",
        "business_line",
        "business line",
        "coverage",
        "product"
    ],
    "company_id": [
        "company_id",
        "company id",
        "company",
        "carrier",
        "insurer",
        "entity"
    ]
}


def standardize_col_names(col_name: str) -> str:
    """
    Standardizes column names from an input CSV into names the application can recognize.
    """
    col_name = col_name.strip().lower()
    col_name = re.sub(r"[_\-]+", " ", col_name)
    col_name = re.sub(r"\s+", " ", col_name)

    return col_name


def map_col_names(df: pd.DataFrame) -> pd.DataFrame:
    col_mapping = {}

    for col in df.columns:
        standardized_col = standardize_col_names(col)

        for standard_name, aliases in STANDARD_COLUMNS.items():
            standardized_aliases = [standardize_col_names(alias) for alias in aliases]

            if standardized_col in standardized_aliases:
                col_mapping[col] = standard_name
                break

    return df.rename(columns=col_mapping)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = map_col_names(df)

    required_cols = ["accident_year", "development_year", "paid_loss"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after cleaning: {missing_cols}")

    cleaned_df = df[required_cols].copy()

    cleaned_df = cleaned_df.dropna(subset=required_cols)

    cleaned_df["accident_year"] = pd.to_numeric(
        cleaned_df["accident_year"],
        errors="coerce"
    )

    cleaned_df["development_year"] = pd.to_numeric(
        cleaned_df["development_year"],
        errors="coerce"
    )

    cleaned_df["paid_loss"] = pd.to_numeric(
        cleaned_df["paid_loss"],
        errors="coerce"
    )

    cleaned_df = cleaned_df.dropna(subset=required_cols)

    cleaned_df["accident_year"] = cleaned_df["accident_year"].astype(int)
    cleaned_df["development_year"] = cleaned_df["development_year"].astype(int)

    # Some uploads use calendar years (for example 2006, 2007) instead of
    # development ages in months (for example 12, 24, 36). The triangle and
    # backtesting code expect month-based development periods, so normalize
    # calendar-year development values to month ages here.
    if cleaned_df["development_year"].ge(1000).all():
        cleaned_df["development_year"] = (
            (cleaned_df["development_year"] - cleaned_df["accident_year"] + 1) * 12
        )

    cleaned_df = cleaned_df[cleaned_df["paid_loss"] >= 0]

    cleaned_df = (
        cleaned_df
        .groupby(["accident_year", "development_year"], as_index=False)
        .agg({"paid_loss": "sum"})
        .sort_values(["accident_year", "development_year"])
        .reset_index(drop=True)
    )

    return cleaned_df