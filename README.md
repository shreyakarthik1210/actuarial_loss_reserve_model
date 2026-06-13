# Actuarial Loss Reserve Forecasting Model

An interactive Streamlit dashboard for estimating insurance claim reserves using the Chain-Ladder method, Monte Carlo simulation, and a BiLSTM machine learning benchmark.

The goal of this project is to combine actuarial reserving concepts with modern data science and machine learning techniques in an interactive Streamlit dashboard.


## Features

- Upload claims data and build cumulative paid loss development triangles
- Calculate age-to-age development factors and project ultimate losses
- Estimate unpaid reserves using the Chain-Ladder method
- Run Monte Carlo simulations to quantify reserve uncertainty
- Train an experimental BiLSTM model for ultimate loss prediction
- Backtest Chain-Ladder and BiLSTM projections against known ultimates using historical calendar-year cutoffs
- Compare model performance via MAE, RMSE, MAPE, and Bias


## Actuarial Concept

The Chain-Ladder method estimates unpaid insurance claim reserves by using historical loss development patterns. It calculates development factors between reporting periods, applies those factors to the latest cumulative losses, and projects each accident year to an estimated ultimate loss.

The reserve for each accident year is calculated as:

```text
Reserve = Projected Ultimate Loss - Latest Cumulative Loss
```

The total reserve is the sum of estimated reserves across all accident years.

## Machine Learning Component

The project includes an experimental Bidirectional Long Short-Term Memory, or BiLSTM, model for ultimate loss prediction. The model uses early cumulative paid loss development as a sequential input and predicts the ultimate loss for each accident year.

This model is intended as a benchmark against the traditional Chain-Ladder method rather than a replacement for actuarial judgment. Since loss triangles are often small, the BiLSTM model should be interpreted carefully and evaluated through backtesting.

The BiLSTM approach was inspired by the following CAS article:
[Ultimate Loss Reserve Forecasting Using Bidirectional LSTMs](https://forum.casact.org/article/37953-ultimate-loss-reserve-forecasting-using-bidirectional-lstms)


## Backtesting

The application includes backtesting functionality for both the Chain-Ladder method and the BiLSTM model. A user can select a calendar stop year, hide later development data, and evaluate how well each method would have predicted known ultimate losses.

The following performance metrics are calculated:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- Bias: Average signed prediction error


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

- PostgreSQL storage for claims data and model run results
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
