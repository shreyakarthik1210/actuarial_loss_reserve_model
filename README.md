# Actuarial Loss Reserve Forecasting Model

This project is an actuarial loss reserving application that estimates insurance
claim reserves using traditional reserving methods, simulation, and an experimental
machine learning benchmark. The application allows users to upload claims data,
build cumulative paid loss development triangles, estimate reserves using the
Chain-Ladder method, run Monte Carlo simulations for reserve uncertainty, and
compare results against a BiLSTM-based ultimate loss prediction model.

The goal of this project is to combine actuarial reserving concepts with modern data science and machine learning techniques in an interactive Streamlit dashboard.


## Features

- Upload insurance claims data
- Build cumulative paid loss development triangles
- Calculate age-to-age loss development factors
- Estimate projected ultimate losses by accident year
- Estimate unpaid reserves using the Chain-Ladder method
- Display total reserve estimates in an interactive Streamlit dashboard
- Run Monte Carlo simulations to analyze reserve uncertainty
- Visualize simulated reserve distributions
- Backtest Chain-Ladder reserve projections using historical calendar-year cutoffs
- Train an experimental BiLSTM model for ultimate loss prediction
- Backtest BiLSTM projections using the same stop-year logic as Chain-Ladder
- Compare model performance using MAE, RMSE, MAPE, and Bias


## Machine Learning Component

The project includes an experimental Bidirectional Long Short-Term Memory, or BiLSTM, model for ultimate loss prediction. The model uses early cumulative paid loss development as a sequential input and predicts the ultimate loss for each accident year.

This model is intended as a benchmark against the traditional Chain-Ladder method rather than a replacement for actuarial judgment. Since loss triangles are often small, the BiLSTM model should be interpreted carefully and evaluated through backtesting.

The BiLSTM approach was inspired by the following CAS article:
[Ultimate Loss Reserve Forecasting Using Bidirectional LSTMs](https://forum.casact.org/article/37953-ultimate-loss-reserve-forecasting-using-bidirectional-lstms)


## Actuarial Concept

The Chain-Ladder method estimates unpaid insurance claim reserves by using historical loss development patterns. It calculates development factors between reporting periods, applies those factors to the latest cumulative losses, and projects each accident year to an estimated ultimate loss.

The reserve for each accident year is calculated as:

```text
Reserve = Projected Ultimate Loss - Latest Cumulative Loss
```

The total reserve is the sum of estimated reserves across all accident years.

## Backtesting

The application includes backtesting functionality for both the Chain-Ladder method and the BiLSTM model. A user can select a calendar stop year, hide later development data, and evaluate how well each method would have predicted known ultimate losses.

The following performance metrics are calculated:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- Bias: Average signed prediction error

These metrics help compare the accuracy of each reserving approach.


## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- Streamlit
- Plotly
- TensorFlow / Keras
- PostgreSQL
- SQLAlchemy


## Planned Features

- PostgreSQL storage for uploaded claims data
- PostgreSQL storage for model run results
- Ability to compare multiple saved model runs
- Additional reserving methods, such as Bornhuetter-Ferguson or Cape Cod
- Improved validation for uploaded claims data
- Exportable reserve reports


## Running This Project

### 1. Create a Python 3.11 virtual environment

```bash
python3.11 -m venv name_of_venv
```

### 2. Activate the virtual environment

For macOS or Linux:

```bash
source ./name_of_venv/bin/activate
```

For Windows:

```bash
name_of_venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

## Project Purpose

This project was built to demonstrate how actuarial reserving methods can be implemented in Python and extended with simulation, backtesting, and machine learning. It highlights both traditional actuarial techniques and experimental approaches for evaluating reserve uncertainty and predictive performance.