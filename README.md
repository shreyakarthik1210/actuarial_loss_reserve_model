# Actuarial Loss Reserve Forecasting Model

This project estimates insurance loss reserves using actuarial reserving methods and simulation.

## Features

- Upload insurance claims data
- Build cumulative paid loss development triangles
- Estimate reserves using the Chain-Ladder method
- Calculate age-to-age development factors
- Estimate projected ultimate losses by accident year
- Display total reserve estimates in a Streamlit dashboard

## Planned Features

- Monte Carlo simulation for reserve uncertainty
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