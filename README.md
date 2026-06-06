# Actuarial Loss Reserve Forecasting Model

This project estimates insurance loss reserves using actuarial reserving methods and simulation.

## Features

- Upload insurance claims data
- Build cumulative paid loss development triangles
- Estimate reserves using the Chain-Ladder method
- Calculate age-to-age development factors
- Estimate projected ultimate losses by accident year
- Display total reserve estimates in a Streamlit dashboard
- Monte Carlo simulation for reserve uncertainty


## Planned Features

- PostgreSQL storage for claims and model runs
- Backtesting against mature accident years
- LSTM/BiLSTM benchmark inspired by actuarial deep learning research

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- Streamlit
- PostgreSQL
- SQLAlchemy
- Plotly

## Actuarial Concept

The Chain-Ladder method estimates unpaid insurance claim reserves by using historical loss development patterns. It calculates development factors between reporting periods, projects latest cumulative losses to ultimate losses, and estimates reserves as:

Reserve = Projected Ultimate Loss - Latest Cumulative Loss

# Running this Project
1. Create a environment with python 3.11: 
```bash
python3.11 -m venv name_of_venv
```
2. Activate the virtual environment (macOS command): 
```bash
source ./name_of_venv/bin/activate
```
3. Run the streamlit application: 
```bash
streamlit run app/streamlit_app.py
```