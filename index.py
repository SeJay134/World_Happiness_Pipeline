from prefect import flow, task, get_run_logger
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import re

@task(log_prints=True)
def path_csv():
    base = Path.cwd()
    folder_path = base / "csv"
    files = list(folder_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("was not finded csv")
    return files

@task(retries=3, retry_delay_seconds=2)
def read_csv_file(files):
    df = []
    for file in files:
        if file.suffix == ".csv":
            try:
                data = pd.read_csv(file, sep=";", on_bad_lines="skip")
                year = file.stem
                match = re.search(r"\d{4}", year)
                if match:
                    year = match.group()
                    data["Year"] = int(year)
                df.append(data)
            except Exception as e:
                print(f"error: {e}")
    if not df:
        raise ValueError("No valid CSV files were loaded")
    
    return pd.concat(df, ignore_index=True)

