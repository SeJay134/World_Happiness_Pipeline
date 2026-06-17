from prefect import flow, task, get_run_logger
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="app.log",
    filemode="a",
    encoding="utf-8"
)

def get_logger(name: str):
    return logging.getLogger(name)

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

@task(retries=3, retry_delay_seconds=2)
def save_csv(df):
    path = Path("outputs/merged_happiness.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

@task
def log_columns(df):
    logger = get_run_logger()
    logger.info(f"Columns:\n {list(df.columns)}")
    logger.info(f"dtypes:\n {list(df.dtypes)}")
    logger.info(f"len:\n {len(df)}")
    logger.info(f"First 5 rows:\n {df.head()}")
    logger.info(f"Last 5 rows:\n {df.tail()}")

    if df.isna().sum().sum() > 0:
        logger.warning("df has missed values")
    
    if df.empty:
        logger.error("df is empty")

    if df.duplicated().sum() > 0:
        logger.error("df has dublicates")
    return df

