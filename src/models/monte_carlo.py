import numpy as np
import pandas as pd
from scipy.stats import lognorm

from src.models.chain_ladder import cal_development_factors


def cal_indv_development_factors (triangle: pd.DataFrame) -> np.ndarray:
    dev_periods = list(triangle.columns)
    individual_factors = dict()

    for i in range (len(dev_periods) - 1):
        cur_period = dev_periods[i]
        next_period = dev_periods[i + 1]
        valid = triangle[[cur_period, next_period]].dropna()

        ratios = valid[next_period]  / valid[cur_period]
        individual_factors[cur_period] = ratios

    return individual_factors

"""
This function estimates the volatility of development factors based on the loss triangle data.
"""
def estimate_volatility (triangle: pd.DataFrame, default_sigma: float = 0.15) -> pd.DataFrame:
    development_factors = cal_development_factors(triangle)
    individual_factors = cal_indv_development_factors(triangle)
    dev_periods = list(triangle.columns)

    volatility = list()
    for i in range (len(dev_periods) - 1):
        cur_period = dev_periods[i]
        next_period = dev_periods[i + 1]

        factors = individual_factors[cur_period]
        factors = factors[factors > 0]

        chain_ladder_factor = development_factors[i]
        if len(factors) > 1:
            log_factors = np.log(factors) - np.log(chain_ladder_factor)  # Calculate log deviations from the chain ladder factor
            sigma = np.std(log_factors, ddof=1) # Use sample standard deviation (ddof=1) for an unbiased estimate

            if (np.isnan(sigma) or sigma <= 0):
                sigma = default_sigma
        else:
            sigma = default_sigma   # Default sigma = 0.15 when there is insufficient data to estimate volatility
        sigma = float(np.clip(sigma, 0.02, 0.30))
        volatility.append(
            {
                "from_development_period": cur_period,
                "to_development_period": next_period, 
                "chain_ladder_factor": chain_ladder_factor,
                "lognormal_sigma": sigma
            }
        )
    return pd.DataFrame(volatility)

"""
This function simulates future development factors based on the estimated volatility.
"""
def simulate_factors(vol_df: pd.DataFrame, state: np.random.Generator) -> np.ndarray:
    simulated_factors = []

    for _, row in vol_df.iterrows():
        mean_factor = float(row["chain_ladder_factor"])
        sigma = float(row["lognormal_sigma"])

        mu = np.log(mean_factor) - 0.5 * sigma**2

        simulated_factor = lognorm(
            s=sigma,
            scale=np.exp(mu)
        ).rvs(random_state=state)

        lower_bound = max(1.0, mean_factor * 0.75)
        upper_bound = mean_factor * 1.25

        simulated_factor = np.clip(
            simulated_factor,
            lower_bound,
            upper_bound
        )

        simulated_factors.append(simulated_factor)

    return np.array(simulated_factors)

"""
This function implements a Monte Carlo simulation for loss reserve estimation. 
It estimates the volatility of development factors, simulates future development factors using a 
lognormal distribution, and projects ultimate losses and reserves for each simulation run. 
"""
def monte_carlo_simulation (triangle: pd.DataFrame, num_simulations: int = 10_000, 
                            random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = np.random.default_rng(random_seed)
    dev_period = list(triangle.columns)
    vol_df = estimate_volatility(triangle)

    results = list()
    for sim in range (1, num_simulations + 1):
        simulated_factors = simulate_factors(vol_df, state)
        total_projected_ultimate = 0
        total_latest_loss = 0
        total_reserve = 0

        for accident_year, row in triangle.iterrows():
            latest_period = row.last_valid_index()
            latest_loss = row[latest_period]
            latest_position = dev_period.index(latest_period)

            future_factors = simulated_factors[latest_position:]
            similated_ultimate = np.prod(future_factors) if len(future_factors) > 0 else 1.0
            projected_ultimate = latest_loss * similated_ultimate
            reserve = max(projected_ultimate - latest_loss, 0)

            total_projected_ultimate += projected_ultimate
            total_latest_loss += latest_loss
            total_reserve += reserve

        results.append({
            "Simulation_id": sim, 
            "Total_Latest_Paid_Loss": total_latest_loss,
            "Total_Projected_Ultimate_Loss": total_projected_ultimate,
            "Total_Estimated_Reserve": total_reserve
        })
    
    simulation_results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Standard Deviation",
            "5th Percentile",
            "25th Percentile",
            "75th Percentile",
            "95th Percentile",
            "Minimum",
            "Maximum"
        ],
        "Total_Reserve": [
            simulation_results_df['Total_Estimated_Reserve'].mean(),
            simulation_results_df['Total_Estimated_Reserve'].median(),
            simulation_results_df['Total_Estimated_Reserve'].std(),
            simulation_results_df['Total_Estimated_Reserve'].quantile(0.05),
            simulation_results_df['Total_Estimated_Reserve'].quantile(0.25),
            simulation_results_df['Total_Estimated_Reserve'].quantile(0.75),
            simulation_results_df['Total_Estimated_Reserve'].quantile(0.95),
            simulation_results_df['Total_Estimated_Reserve'].min(),
            simulation_results_df['Total_Estimated_Reserve'].max()
        ]
    })

    return simulation_results_df, summary_df, vol_df