import pandas as pd
"""
Validates the claims data before processing. Checks for required columns, null values, negative paid losses, and duplicates.
"""
def validate_claims_data(df: pd.DataFrame) -> list[str]:
    errors = []

    required_cols = {'accident_year', 'development_year', 'paid_loss'}
    missing_cols = required_cols - set(df.columns)

    if (missing_cols):  # Check if any of the required columns are missing
        errors.append(f"Missing required columns: {missing_cols}")
        return errors
    
    if (df[["accident_year", "development_year", "paid_loss"]].isnull().any().any()):   # Check for null values in required columns
        errors.append("Null values found in required columns.")

    if (df["paid_loss"] < 0).any(): # Check for negative paid losses
        errors.append("Paid loss values cannot be negative.")

    duplicates = df.duplicated(subset=['accident_year', 'development_year']).sum()  # Check for duplicates
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate rows based on 'accident_year' and 'development_year'.")
    
    return errors

