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

def get_run_logger(name: str):
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

@task
def clean_and_transform(db):
    logger = get_run_logger()
    db_clean = db.copy()
    db_clean.columns = db_clean.columns.str.strip()
    nan_percent = (db_clean.isna().sum() / len(db_clean) * 100).round(1)
    logger.info(f"nan_percent.sort_values: {nan_percent.sort_values(ascending=False)}")

    cols = ["Happiness score", "GDP per capita", "Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]

    for col in cols:
        db_clean[col] = (db_clean[col].astype(str).str.replace(",", ".", regex=False))

    db_clean["Happiness score"] = pd.to_numeric(db_clean["Happiness score"], errors="coerce")
    db_clean["GDP per capita"] = pd.to_numeric(db_clean["GDP per capita"], errors="coerce")
    db_clean["Social support"] = pd.to_numeric(db_clean["Social support"], errors="coerce")
    db_clean["Freedom to make life choices"] = pd.to_numeric(db_clean["Freedom to make life choices"], errors="coerce")
    db_clean["Generosity"] = pd.to_numeric(db_clean["Generosity"], errors="coerce")
    db_clean["Perceptions of corruption"] = pd.to_numeric(db_clean["Perceptions of corruption"], errors="coerce")
    logger.info(f"db_clean.dtypes:\n {db_clean.dtypes}")

    db_clean = db_clean.drop(columns=["Ladder score"], errors="ignore")

    mode_val = db_clean["Regional indicator"].mode()
    if not mode_val.empty:
        db_clean["Regional indicator"] = db_clean["Regional indicator"].fillna(mode_val[0])

    db_clean["Happiness score"] = db_clean["Happiness score"].fillna(db_clean["Happiness score"].median())
    db_clean["Healthy life expectancy"] = db_clean["Healthy life expectancy"].fillna(db_clean["Healthy life expectancy"].mean())
    logger.info(f"db_clean dtype {db_clean.dtypes}")

    nan_percent_new = (db_clean.isna().sum() / len(db_clean) * 100).round(1)
    logger.info(f"nan_percent_new.sort_values\n {nan_percent_new.sort_values(ascending=False)}")
    return db_clean

@task
def happiness_score_descr_stat(db_clean):
    logger = get_run_logger()
    hap = db_clean["Happiness score"]
    mean_data = np.mean(hap)
    median_data = np.median(hap)
    std_data = np.std(hap)

    logger.info(f"mean_data: {mean_data}")
    logger.info(f"median_data: {median_data}")
    logger.info(f"std_data: {std_data}")

    # mean happiness score grouped by year and by region.
    year_group_mean = db_clean.groupby("Year")["Happiness score"].mean().sort_index()
    logger.info(f"year_group_mean:\n {year_group_mean}")

    reg_group_mean = db_clean.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    logger.info(f"reg_group_mean:\n {reg_group_mean}")

    logger.info(f"Top region: {reg_group_mean.idxmax()} ({reg_group_mean.max()})")
    logger.info(f"Worst region: {reg_group_mean.idxmin()} ({reg_group_mean.min()})")

    return mean_data, median_data, std_data, year_group_mean, reg_group_mean

@task
def plot_hist(db_clean):
    logger = get_run_logger()
    path = Path("outputs/happiness_histogram.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(db_clean["Happiness score"], bins=30, edgecolor="black")
    plt.title("All happiness scores across all years")
    plt.xlabel("Happiness score")
    plt.ylabel("Frequency")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info(f"plot_hist")
    return path

@task
def plot_box(db_clean):
    logger = get_run_logger()
    path = Path("outputs/happiness_by_year.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data = db_clean, x = "Year", y = "Happiness score")
    plt.title("Happiness score distributions across years")
    plt.xlabel("Year")
    plt.ylabel("Happiness score")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info(f"plot_box")
    return path

@task
def gdp_vs_happ(db_clean):
    logger = get_run_logger()
    path = Path("outputs/gdp_vs_happiness.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=db_clean, x = "GDP per capita", y = "Happiness score")
    plt.title("The relationship between GDP per capita and happiness score")
    plt.xlabel("GDP")
    plt.ylabel("Happiness score")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info(f"gdp_vs_happ")
    return path

@task
def corr_heatmap(db_clean):
    logger = get_run_logger()
    corr_matrix = db_clean.corr(numeric_only=True)
    path = Path("outputs/correlation_heatmap.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(path, dpi=300)
    plt.close()

    logger.info(f"corr_heatmap")
    return path

@task
def stats_test(db_clean):
    logger = get_run_logger()
    hap_score_2019 = db_clean[db_clean["Year"] == 2019]["Happiness score"]
    hap_score_2020 = db_clean[db_clean["Year"] == 2020]["Happiness score"]
    t_stat, p_stat = stats.ttest_ind(hap_score_2019, hap_score_2020, equal_var=False)
    mean_2019 = hap_score_2019.mean()
    mean_2020 = hap_score_2020.mean()
    logger.info(f"Mean happiness 2019: {mean_2019:.3f}, Mean happiness 2020: {mean_2020:.3f}")

    if p_stat < 0.05:
        if mean_2019 > mean_2020:
            logger.info(f"Happiness scores decreased significantly from 2019 to 2020.")
        else:
            logger.info(f"Happiness scores increased significantly from 2019 to 2020.")
    else:
        logger.info(f"No statistically significant change in happiness scores between 2019 and 2020.")

    logger.info(f"t_stat: {t_stat:.5F}, p_stat: {p_stat:.5F}")
    return t_stat, p_stat, mean_2019, mean_2020

