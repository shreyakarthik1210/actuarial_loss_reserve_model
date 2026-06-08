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
from src.data.validator import validate_claims_data
from src.metrics.backtesting import backtest
from src.metrics.backtesting import backtest_loss_triangle
from src.models.bilstm_reserving import training_data, train_bilstm_model, predict_ultimates_bilstm


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
chainladder, montecarlo, bilstm = st.tabs(["Chain-Ladder Method", "Monte Carlo Simulation", "BiLSTM Reserving"])

try:
    # Validate the uploaded claims data and display any errors
    errors = validate_claims_data(claims_df)
    if errors:
        st.error("Data validation failed: ")
        count = 1
        for error in errors: 
            st.write(f"{count}. {error}")
            count += 1

    triangle = loss_triangle_builder(claims_df) # Convert raw claims data to loss triangle format

    # Chain-ladder method implementation and backtesting
    with chainladder:
        st.subheader("Cumulative Paid Loss Triangle")
        st.dataframe(triangle)

        development_factors, cdfs, reserve_results = chain_ladder(triangle)

        st.subheader("Development Factors")
        st.write("Development factors are calculated as the ratio of losses in the next development period to the current period, aggregated across all accident years. ")
        factor_df = pd.DataFrame({
            "from_development_period": triangle.columns[:-1],
            "to_development_period": triangle.columns[1:],
            "development_factor": development_factors
        })
        st.dataframe(factor_df)

        st.subheader("CDFs to Ultimate")
        st.write("Cumulative Distribution Functions (CDFs) represent the proportion of ultimate losses that have been paid by each development period.")
        cdf_df = pd.DataFrame({
            "development_period": triangle.columns,
            "cdf_to_ultimate": cdfs
        })
        st.dataframe(cdf_df)

        st.subheader("Chain-Ladder Reserve Estimates")
        st.write("The table below shows the projected ultimate loss and estimated reserve for each accident year based on the Chain-Ladder method.")
        st.dataframe(reserve_results)

        total_reserve = reserve_results["estimated_reserve"].sum()
        total_ultimate = reserve_results["projected_ultimate_loss"].sum()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Projected Ultimate Loss", f"${total_ultimate:,.0f}")

        with col2:
            st.metric("Total Estimated Reserve", f"${total_reserve:,.0f}")

        st.space(size="medium")
        st.subheader("Backtesting Chain-Ladder Projections")
        st.write("Backtesting allows us to evaluate the accuracy of our Chain-Ladder projections by comparing them to actual observed losses. " \
            "Select a stop year to run the backtest.")
        st.space(size="small")
        development_periods = sorted([int(col) for col in triangle.columns])
        stop_year = st.slider(
            "Select calendar stop year for backtesting",
            min_value=int(triangle.index.min()),
            max_value=int(triangle.index.max()),
            value=int(triangle.index.max()) - 1,
            step=1
        )
        
        if (st.button("Run Chain-Ladder Backtests")):
            debug_triangle = backtest_loss_triangle(triangle, stop_year)
            debug_triangle = debug_triangle.dropna(axis=0, how="all")
            debug_triangle = debug_triangle.dropna(axis=1, how="all")

            st.subheader("Backtesting Paid Loss Triangle")
            st.dataframe(debug_triangle)

            backtest_results, backtest_errors = backtest(triangle, stop_year)

            st.subheader("Backtesting Results")
            st.dataframe(backtest_results)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"${backtest_errors['MAE']:,.0f}")
            with col2:
                st.metric("RMSE", f"${backtest_errors['RMSE']:,.0f}")
            with col3:
                st.metric("MAPE", f"{backtest_errors['MAPE']:.2f}%")
            with col4:
                st.metric("Bias", f"${backtest_errors['Bias']:,.0f}")

    # Monte carlo simulation implementation 
    with montecarlo:
        st.subheader("Monte Carlo Reserve Simulation")
        st.write("This section simulates future development factors based on the estimated volatility of historical development factors with the Monte Carlo method. ")

        st.subheader("Cumulative Paid Loss Triangle")
        st.dataframe(triangle)

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

            st.plotly_chart(fig, width='stretch')

            st.subheader("Raw Simulation Results")
            st.write(f"Showing all {len(simulation_results):,} simulation runs.")
            st.dataframe(simulation_results)
        
    with bilstm:
        st.subheader("BiLSTM Ultimate Loss Prediction")
        st.write(
            "This model uses early cumulative paid loss development as a sequence "
            "and predicts ultimate loss. It is an experimental benchmark against "
            "the traditional Chain-Ladder model."
        )

        length = st.selectbox(
            "Select the number of development periods to use as input features for the BiLSTM model:",
            options=list(range(1, len(triangle.columns))),
            index=0,
            key="bilstm_length"
        )

        bilstm_epochs = st.slider(
            "Training epochs",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="bilstm_epochs"
        )

        bilstm_batch_size = st.select_slider(
            "Batch size",
            options=[8, 16, 32, 64, 128, 256],
            value=64,
            key="bilstm_batch_size"
        )

        # Initialize session state
        if "bilstm_model" not in st.session_state:
            st.session_state.bilstm_model = None

        if "bilstm_history" not in st.session_state:
            st.session_state.bilstm_history = None

        if "bilstm_predictions" not in st.session_state:
            st.session_state.bilstm_predictions = None

        if "bilstm_scale" not in st.session_state:
            st.session_state.bilstm_scale = None

        train_clicked = st.button("Train BiLSTM Model", key="train_bilstm_button")

        if train_clicked:
            try:
                with st.spinner("Preparing BiLSTM training data..."):
                    X, y, scale = training_data(
                        t=triangle,
                        length=length
                    )

                st.write(f"Training examples: {X.shape[0]}")
                st.write(f"Input shape: {X.shape}")

                if X.shape[0] < 2:
                    st.warning(
                        "There are very few training examples. "
                        "The BiLSTM may not train meaningfully on this triangle."
                    )

                with st.spinner("Training BiLSTM model..."):
                    model, history = train_bilstm_model(
                        X=X,
                        y=y,
                        length=length,
                        epochs=bilstm_epochs,
                        batch_size=bilstm_batch_size,
                        verbose=1
                    )

                with st.spinner("Generating predictions..."):
                    bilstm_predictions = predict_ultimates_bilstm(
                        model=model,
                        t=triangle,
                        length=length,
                        scale=scale
                    )

                st.session_state.bilstm_model = model
                st.session_state.bilstm_history = history
                st.session_state.bilstm_predictions = bilstm_predictions
                st.session_state.bilstm_scale = scale

                st.success("BiLSTM model trained successfully.")

            except Exception as e:
                st.error(f"BiLSTM Error: {e}")

        if st.session_state.bilstm_predictions is not None:
            st.subheader("BiLSTM Ultimate Loss Predictions")
            st.dataframe(st.session_state.bilstm_predictions)

        if st.session_state.bilstm_history is not None:
            st.subheader("Training Loss")
            loss_df = pd.DataFrame({
                "loss": st.session_state.bilstm_history.history["loss"]
            })
            st.line_chart(loss_df)
except Exception as error:
    st.error(f"Error: {error}")