import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.triangle_builder import loss_triangle_builder
from src.models.chain_ladder import chain_ladder
from src.models.monte_carlo import monte_carlo_simulation


st.set_page_config(
    page_title="Actuarial Loss Reserve Forecasting",
    layout="wide"
)

st.title("Actuarial Loss Reserve Forecasting Model")

st.write(
    """
    This dashboard estimates insurance loss reserves using the Chain-Ladder method.
    Future versions will include Monte Carlo simulation, PostgreSQL storage, and LSTM benchmarking.
    """
)

uploaded_file = st.file_uploader(
    "Upload claims data CSV",
    type=["csv"]
)

if uploaded_file is not None:
    claims_df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample claims data.")
    claims_df = pd.read_csv(ROOT_DIR / "data" / "sample_claims.csv")

st.subheader("Raw Claims Data")
st.dataframe(claims_df)

try:
    triangle = loss_triangle_builder(claims_df)

    st.subheader("Cumulative Paid Loss Triangle")
    st.dataframe(triangle)

    development_factors, cdfs, reserve_results = chain_ladder(claims_df)

    st.subheader("Development Factors")
    factor_df = pd.DataFrame({
        "from_development_period": triangle.columns[:-1],
        "to_development_period": triangle.columns[1:],
        "development_factor": development_factors
    })
    st.dataframe(factor_df)

    st.subheader("CDFs to Ultimate")
    cdf_df = pd.DataFrame({
        "development_period": triangle.columns,
        "cdf_to_ultimate": cdfs
    })
    st.dataframe(cdf_df)

    st.subheader("Chain-Ladder Reserve Estimates")
    st.dataframe(reserve_results)

    total_reserve = reserve_results["estimated_reserve"].sum()
    total_ultimate = reserve_results["projected_ultimate_loss"].sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Projected Ultimate Loss", f"${total_ultimate:,.0f}")

    with col2:
        st.metric("Total Estimated Reserve", f"${total_reserve:,.0f}")

    st.subheader("Monte Carlo Reserve Simulation")

    n_simulations = st.slider(
        "Number of simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )

    run_simulation = st.button("Run Monte Carlo Simulation")

    if run_simulation:
        with st.spinner("Running Monte Carlo simulation..."):
            simulation_results, simulation_summary, volatility_df = monte_carlo_simulation(
                triangle=triangle,
                num_simulations=n_simulations,
                random_seed=42
            )

        st.subheader("Development Factor Volatility Assumptions")
        st.dataframe(volatility_df)

        st.subheader("Monte Carlo Reserve Summary")
        st.dataframe(simulation_summary)

        mean_reserve = simulation_results["Total_Estimated_Reserve"].mean()
        median_reserve = simulation_results["Total_Estimated_Reserve"].median()
        p95_reserve = simulation_results["Total_Estimated_Reserve"].quantile(0.95)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Simulated Reserve", f"${mean_reserve:,.0f}")

        with col2:
            st.metric("Median Simulated Reserve", f"${median_reserve:,.0f}")

        with col3:
            st.metric("95th Percentile Reserve", f"${p95_reserve:,.0f}")

        fig = px.histogram(
            simulation_results,
            x="Total_Estimated_Reserve",
            nbins=50,
            title="Monte Carlo Simulated Total Reserve Distribution"
        )

        fig.update_layout(
            xaxis_title="Total Reserve",
            yaxis_title="Frequency"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Simulation Results")
        st.write(f"Showing all {len(simulation_results):,} simulation runs.")
        st.dataframe(simulation_results)

except Exception as error:
    st.error(f"Error: {error}")