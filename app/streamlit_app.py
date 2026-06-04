import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.triangle_builder import loss_triangle_builder
from src.models.chain_ladder import chain_ladder


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

except Exception as error:
    st.error(f"Error: {error}")