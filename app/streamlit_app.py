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
from src.metrics.backtesting import backtest_loss_triangle, cal_errors
from src.models.bilstm_reserving import training_data, train_bilstm_model, predict_ultimates_bilstm
from src.models.bilstm_reserving import bilstm_predictions_to_triangle, bilstm_predictions_to_pred_df
from data.data_cleaner import clean_data



st.set_page_config(
    page_title="Actuarial Loss Reserve Forecasting",
    layout="wide"
)



st.title("Actuarial Loss Reserve Forecasting Model")

st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        border: none;
        box-shadow: none;
        padding-top: 0;
        padding-bottom: 0;
    }
    div[data-testid="stSlider"] {
        border: none;
        box-shadow: none;
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This dashboard estimates insurance loss reserves using two approaches:

    - **Chain-Ladder** for deterministic reserve projections, with **Monte Carlo simulation** to model uncertainty.
    - **BiLSTM** as an experimental benchmark that uses early development patterns to predict ultimate losses.

    Both methods can be backtested against observed losses by selecting a calendar stop year.
    The **Compare Methods** tab shows the Chain-Ladder and BiLSTM backtesting results side by side.
    """
)
st.space(size="small")
uploaded_file = st.file_uploader(
    "Upload claims data CSV (if no file is uploaded, a sample dataset will be used)",
    type=["csv"]
)

if uploaded_file is not None:
    claims_df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample claims data.")
    claims_df = pd.read_csv(ROOT_DIR / "data" / "sample_claims.csv")

claims_df = clean_data(claims_df) 
st.subheader("Cleaned Raw Claims Data")
with st.expander("Click to collapse data", expanded=True):
    st.dataframe(claims_df)
chainladder, montecarlo, bilstm, compare_methods = st.tabs(["Chain-Ladder Method", "Monte Carlo Simulation", "BiLSTM Reserving", "Compare Methods"])

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
        st.subheader("Chain-Ladder Method")
        st.write("The Chain-Ladder method estimates ultimate losses and reserves based on historical development patterns in the loss triangle. "
            "It calculates development factors and applies them to project future losses.")
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
        url = "https://forum.casact.org/article/37953-ultimate-loss-reserve-forecasting-using-bidirectional-lstms"
        st.caption("The BiLSTM model is experimental and based on a simplified version of this [CAS paper.](%s)" % url)
        st.space(size="small")
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


        st.subheader("Backtesting BiLSTM Projections")
        st.write("Run backtesting for the BiLSTM model.")

        stop_year_bilstm = st.slider(
            "Select calendar stop year for BiLSTM backtesting",
            min_value=int(triangle.index.min()),
            max_value=int(triangle.index.max()),
            value=int(triangle.index.max()) - 1,
            step=1,
            key="bilstm_stop_year"
        )

        if st.button("Run BiLSTM Backtests"):
            if st.session_state.bilstm_predictions is None:
                st.warning("No BiLSTM predictions available — train the model first.")
            else:
                # Build the triangle as of the stop year
                debug_triangle = backtest_loss_triangle(triangle, stop_year_bilstm)
                debug_triangle = debug_triangle.dropna(axis=0, how="all")
                debug_triangle = debug_triangle.dropna(axis=1, how="all")

                # Insert BiLSTM predicted ultimates into the truncated triangle
                bilstm_tri = bilstm_predictions_to_triangle(
                    st.session_state.bilstm_predictions,
                    debug_triangle
                )

                st.subheader("Backtesting Paid Loss Triangle (with BiLSTM ultimates)")
                st.dataframe(bilstm_tri)

                # Build results comparing BiLSTM predicted ultimates to actuals
                actual = triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)

                results = []
                for _, row in st.session_state.bilstm_predictions.iterrows():
                    ay = row["accident_year"]
                    if ay not in actual.index:
                        continue
                    predicted_ultimate_loss = row["bilstm_predicted_ultimate_loss"]
                    actual_ultimate_loss = actual.loc[ay]
                    results.append({
                        "accident_year": ay,
                        "predicted_ultimate_loss": predicted_ultimate_loss,
                        "actual_ultimate_loss": actual_ultimate_loss,
                    })

                results_df = pd.DataFrame(results)

                st.subheader("BiLSTM Backtesting Results")
                st.dataframe(results_df)

                errors = cal_errors(results_df)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"${errors['MAE']:,.0f}")
                with col2:
                    st.metric("RMSE", f"${errors['RMSE']:,.0f}")
                with col3:
                    st.metric("MAPE", f"{errors['MAPE']:.2f}%")
                with col4:
                    st.metric("Bias", f"${errors['Bias']:,.0f}")
    with compare_methods:
        st.subheader("Method Comparison")
        st.write("Compare the backtesting results for Chain-Ladder and BiLSTM methods")
        st.caption("BiLSTM comparison retrains on submit using the same length, epochs, and batch size selected in the BiLSTM tab.")

        compare_length = length

        with st.form("compare_methods_form", border=False):
            compare_stop_year_input = st.slider(
                "Select calendar stop year for method comparison",
                min_value=int(triangle.index.min()),
                max_value=int(triangle.index.max()),
                value=int(triangle.index.max()) - 1,
                step=1,
            )
            submitted = st.form_submit_button("Run Comparison")

        if submitted:
            with st.spinner("Running comparison..."):
                try:
                    compare_stop_year = compare_stop_year_input
                    comparison_triangle = backtest_loss_triangle(triangle, compare_stop_year)
                    comparison_triangle = comparison_triangle.dropna(axis=0, how="all")
                    comparison_triangle = comparison_triangle.dropna(axis=1, how="all")

                    chain_development_factors, chain_cdfs, chain_results = chain_ladder(comparison_triangle)
                    chain_backtest_df, chain_errors = backtest(triangle, compare_stop_year)

                    compare_epochs = bilstm_epochs
                    compare_batch_size = bilstm_batch_size

                    with st.spinner("Preparing BiLSTM training data..."):
                        X_compare, y_compare, compare_scale = training_data(
                            t=triangle,
                            length=compare_length,
                        )

                    if X_compare.shape[0] < 2:
                        st.warning(
                            "The BiLSTM comparison has very few training examples. "
                            "Results may not be meaningful on this triangle."
                        )

                    with st.spinner("Training BiLSTM model for comparison..."):
                        compare_model, _ = train_bilstm_model(
                            X=X_compare,
                            y=y_compare,
                            length=compare_length,
                            epochs=compare_epochs,
                            batch_size=compare_batch_size,
                            verbose=0,
                        )

                    with st.spinner("Generating BiLSTM comparison predictions..."):
                        bilstm_predictions = predict_ultimates_bilstm(
                            model=compare_model,
                            t=triangle,
                            length=compare_length,
                            scale=compare_scale,
                        )

                    bilstm_backtest_triangle = bilstm_predictions_to_triangle(
                        bilstm_predictions,
                        comparison_triangle,
                    )

                    actual_ultimate = triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
                    comparison_rows = []
                    for accident_year in sorted(set(chain_results["accident_year"]).intersection(bilstm_predictions["accident_year"])):
                        chain_value = float(chain_results.loc[chain_results["accident_year"] == accident_year, "projected_ultimate_loss"].iloc[0])
                        bilstm_value = float(
                            bilstm_predictions.loc[
                                bilstm_predictions["accident_year"] == accident_year,
                                "bilstm_predicted_ultimate_loss",
                            ].iloc[0]
                        )
                        actual_value = float(actual_ultimate.loc[accident_year])
                        comparison_rows.append(
                            {
                                "accident_year": accident_year,
                                "actual_ultimate_loss": actual_value,
                                "chain_ladder_predicted_ultimate_loss": chain_value,
                                "bilstm_predicted_ultimate_loss": bilstm_value,
                                "chain_ladder_abs_error": abs(chain_value - actual_value),
                                "bilstm_abs_error": abs(bilstm_value - actual_value),
                            }
                        )

                    comparison_df = pd.DataFrame(comparison_rows)
                    chain_compare_metrics = cal_errors(
                        comparison_df.rename(columns={"chain_ladder_predicted_ultimate_loss": "predicted_ultimate_loss"})[
                            ["accident_year", "predicted_ultimate_loss", "actual_ultimate_loss"]
                        ]
                    )
                    bilstm_compare_metrics = cal_errors(
                        comparison_df.rename(columns={"bilstm_predicted_ultimate_loss": "predicted_ultimate_loss"})[
                            ["accident_year", "predicted_ultimate_loss", "actual_ultimate_loss"]
                        ]
                    )

                    st.subheader(f"Comparison Results for Year {compare_stop_year}")
                    st.dataframe(comparison_df)

                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.markdown("**Chain-Ladder Backtest**")
                        st.metric("MAE", f"${chain_compare_metrics['MAE']:,.0f}")
                        st.metric("RMSE", f"${chain_compare_metrics['RMSE']:,.0f}")
                        st.metric("MAPE", f"{chain_compare_metrics['MAPE']:.2f}%")
                        st.metric("Bias", f"${chain_compare_metrics['Bias']:,.0f}")
                    with metric_cols[1]:
                        st.markdown("**BiLSTM Backtest**")
                        st.metric("MAE", f"${bilstm_compare_metrics['MAE']:,.0f}")
                        st.metric("RMSE", f"${bilstm_compare_metrics['RMSE']:,.0f}")
                        st.metric("MAPE", f"{bilstm_compare_metrics['MAPE']:.2f}%")
                        st.metric("Bias", f"${bilstm_compare_metrics['Bias']:,.0f}")

                    chart_df = comparison_df.melt(
                        id_vars=["accident_year"],
                        value_vars=[
                            "actual_ultimate_loss",
                            "chain_ladder_predicted_ultimate_loss",
                            "bilstm_predicted_ultimate_loss",
                        ],
                        var_name="series",
                        value_name="ultimate_loss",
                    )
                    chart_df["series"] = chart_df["series"].replace(
                        {
                            "actual_ultimate_loss": "Actual Ultimate",
                            "chain_ladder_predicted_ultimate_loss": "Chain-Ladder Expected",
                            "bilstm_predicted_ultimate_loss": "BiLSTM Expected",
                        }
                    )

                    fig = px.bar(
                        chart_df,
                        x="accident_year",
                        y="ultimate_loss",
                        color="series",
                        barmode="group",
                        title="Actual vs Expected Ultimate Loss by Method",
                        labels={
                            "accident_year": "Accident Year",
                            "ultimate_loss": "Ultimate Loss",
                            "series": "Series",
                        },
                    )
                    fig.update_layout(legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        "Expected value = each method's projected ultimate loss; resulting value = the observed actual ultimate loss."
                    )

                    st.caption(
                        f"BiLSTM comparison trained automatically with {compare_epochs} epochs and batch size {compare_batch_size}."
                    )

                except Exception as error:
                    st.error(f"Comparison error: {error}")
        else:
            st.info("Select a calendar year and click Run Comparison to load the results.")
        
except Exception as error:
    st.error(f"Error: {error}")