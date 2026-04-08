# prefect_warmup.py

import numpy as np
import pandas as pd
from prefect import task, flow


#  Convert to pandas Series
@task
def create_series(arr):
    return pd.Series(arr, name="values")

# Clean data by removing NaNs
@task
def clean_data(series):
    return series.dropna()

#  Summarize the cleaned data
@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary


# Prefect flow connecting tasks
@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    print("Pipeline Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    pipeline_flow()

