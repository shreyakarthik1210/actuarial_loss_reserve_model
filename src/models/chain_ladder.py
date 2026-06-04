import pandas as pd 
import numpy as np
from src.data.triangle_builder import loss_triangle_builder



"""
This function calculates the development factors for each development period based on the loss triangle.
The development factor for a period is calculated as the sum of losses in the next period divided by 
the sum of losses in the current period, using only the rows where both periods have valid data.
"""
def cal_development_factors (triangle: pd.DataFrame) -> np.ndarray:
    cols = list(triangle.columns)
    factors = []

    for i in range (len(cols) - 1):
        cur = cols[i]
        next = cols[i + 1]

        # Get rid of rows with NaN
        valid = triangle[[cur, next]].dropna()

        if valid.empty:
            raise ValueError (
                f"Development factor cannot be calculated for column '{cur}' as it contains only NaN values."
            )

        # Using formula: sum of next period losses / sum of current period losses
        factor = valid[next].sum() / valid[cur].sum()
        factors.append(factor)

    return np.array(factors)


"""
Implements the Chain-Ladder method for loss reserve estimation. 
This is a deterministic method that calculates development factors and applies them to the loss triangle to project future losses.
"""
def chain_ladder(t: pd.DataFrame) -> pd.DataFrame:

    """
    This function calculates the cumulative development factors (CDFs) to ultimate loss for each development period.
    The CDF for a period is the product of all development factors from that period to the ultimate period. 
    """
    def cal_cdfs (factors: np.ndarray) -> np.ndarray:
        cdfs = list()
        for i in range (len(factors)):
            cdf = np.prod(factors[i:])  # CDF is the product of all factors from the current period to the end
            cdfs.append(cdf)
        
        cdfs.append(1.0)

        return np.array(cdfs)

    df = loss_triangle_builder(t) # Convert raw claims data to loss triangle format
    development_periods = list(df.columns)
    development_factors = cal_development_factors(df)
    cdfs = cal_cdfs(development_factors)

    outputs = list()
    for accident_year, row in df.iterrows():
        valid_row = row.dropna()
        if (valid_row.empty):
            continue

        latest_period = valid_row.index[-1]
        latest_loss = valid_row[latest_period]
        latest_position = development_periods.index(latest_period)
        ultimate_loss = latest_loss * cdfs[latest_position] # Project ultimate loss using the CDF for the latest development period
        reserve = ultimate_loss - latest_loss # Calculate reserve as the difference between projected ultimate loss and latest observed loss

        outputs.append({
            "accident_year": accident_year,
            "latest_development_period": latest_period,
            "latest_paid_loss": latest_loss,
            "cdf_to_ultimate": cdfs[latest_position],
            "projected_ultimate_loss": ultimate_loss,
            "estimated_reserve": reserve
        })

    result = pd.DataFrame(outputs)
    return development_factors, cdfs, result



#print(chain_ladder(pd.read_csv("data/sample_claims.csv")))