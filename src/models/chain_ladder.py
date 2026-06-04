import pandas as pd 
import numpy as np
from src.data.triangle_builder import loss_triangle_builder

def chain_ladder(t: pd.DataFrame) -> pd.DataFrame:

    def cal_development_factors (triangle: pd.DataFrame) -> np.ndarray:
        cols = list(triangle.columns)
        factors = []

        for i in range (len(cols) - 1):
            cur = cols[i]
            next = cols[i + 1]

            valid = triangle[[cur, next]].dropna()

            if valid.empty:
                raise ValueError (
                    f"Development factor cannot be calculated for column '{cur}' as it contains only NaN values."
                )

            factor = valid[next].sum() / valid[cur].sum()
            factors.append(factor)

        return np.array(factors)

    def cal_cdfs (factors: np.ndarray) -> np.ndarray:
        cdfs = list()
        for i in range (len(factors)):
            cdf = np.prod(factors[i:])
            cdfs.append(cdf)
        
        cdfs.append(1.0)

        return np.array(cdfs)

    df = loss_triangle_builder(t)
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
        ultimate_loss = latest_loss * cdfs[latest_position]
        reserve = ultimate_loss - latest_loss

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





print(chain_ladder(pd.read_csv("data/sample_claims.csv")))