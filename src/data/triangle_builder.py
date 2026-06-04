import pandas as pd

"""
Constructs a cumulative loss triangle for the Claim Ladder method given a DataFrame.
"""
def loss_triangle_builder (df: pd.DataFrame)-> pd.DataFrame:
    req_cols = ['accident_year', 'development_year', 'paid_loss']
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"DataFrame must contain the following columns (triangle): {req_cols}")
    
    triangle = df.pivot(
        index='accident_year', 
        columns='development_year', 
        values='paid_loss'
    )

    triangle = triangle.sort_index().sort_index(axis=1)
    return triangle

#print(loss_triangle_builder(pd.read_csv("data/sample_claims.csv")))