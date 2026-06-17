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

@task
def stats_second_test(db_clean):
    logger = get_run_logger()
    db_year = db_clean[db_clean["Year"].isin([2019, 2020])]
    west_eur = db_year[db_year["Regional indicator"] == "Western Europe"]["Social support"].dropna()
    # logger.info(f"west_eur: {west_eur[:5]}")
    sub_suh_afr = db_year[db_year["Regional indicator"] == "Sub-Saharan Africa"]["Social support"].dropna()
    # logger.info(f"sub_suh_afr: {sub_suh_afr[:5]}")

    t_stat, p_stat = stats.ttest_ind(west_eur, sub_suh_afr, equal_var=False)
    logger.info(f"t-statistic: {t_stat:.5f} p-value: {p_stat:.2e}")

    mean_west = west_eur.mean()
    mean_sub = sub_suh_afr.mean()

    if p_stat < 0.05:
        if mean_west > mean_sub:
            logger.info(f"Social support in Western Europe is significantly higher than in Sub-Saharan Africa (2019–2020).")
        else:
            logger.info(f"Social support in Sub-Saharan Africa is significantly higher than in Western Europe (2019–2020).")
    else:
        logger.info(f"No statistically significant difference in social support between Western Europe and Sub-Saharan Africa (2019–2020).")

    return t_stat, p_stat

@task
def value_pearson(db_clean):
    logger = get_run_logger()

    y = db_clean["Happiness score"] # Happiness score
    int_col = db_clean.select_dtypes(include="number").columns # columns int

    number_of_tests = 0 # total amount tests
    summary = [] # col, corr, p_value

    for col in int_col:
        if col == "Happiness score" or col == "Year" or col == "Ranking": # came out
            continue
            
        corr, p_value = pearsonr(db_clean[col], y)
        logger.info(f"corr: {col} vs Happiness score")
        logger.info(f"  correlation: {corr:.3f}, p-value: {p_value:.2e}")

        number_of_tests += 1 # more one test
        summary.append((col, corr, p_value)) # add

    # logger.info(f"summary {summary[:5]}")
        
    logger.info(f"number_of_tests: {number_of_tests}")
    adjusted_alpha = 0.05 / number_of_tests             # adjastment Bonferroni correction
    logger.info(f"adjusted_alpha {adjusted_alpha:.5f}")

    for col, corr, p_value in summary:
        if p_value < adjusted_alpha:
            logger.info(f"{col} is significant after Bonferroni correction")
        elif p_value < 0.05:
            logger.info(f"{col} is significant at alpha=0.05 but not after correction")
        else:
            logger.info(f"{col} is not significant")
    
    return summary, adjusted_alpha

@task
def summary_report(db_clean, adjusted_alpha, t_test_result, pearson_summary):
    logger = get_run_logger()

    # Total number of countries and years in the merged dataset.
    total_number_of_countries = db_clean["Country"].nunique()
    logger.info(f"total_number_of_countries: {total_number_of_countries}")

    total_years = db_clean["Year"].nunique()
    logger.info(f"total_years: {total_years}")

    # The top 3 and bottom 3 regions by mean happiness score.
    top_3_regions_by_mean_happiness_score = db_clean.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False).head(3)
    logger.info(f"top_3_regions_by_mean_happiness_score: {top_3_regions_by_mean_happiness_score}")

    bottom_3_regions_by_mean_happiness_score = db_clean.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=True).head(3)
    logger.info(f"bottom_3_regions_by_mean_happiness_score: {bottom_3_regions_by_mean_happiness_score}")

    # The result of the pre/post-2020 t-test in plain language.
    t_stat, p_stat, mean_2019, mean_2020 = t_test_result
    if p_stat < 0.05:
        if mean_2020 > mean_2019:
            logger.info("Happiness scores increased significantly from 2019 to 2020.")
        else:
            logger.info("Happiness scores decreased significantly from 2019 to 2020.")
    else:
        logger.info("No statistically significant change in happiness scores between 2019 and 2020.")

    # The variable most strongly correlated with happiness score (after Bonferroni correction).
    significant_vars = []
    for col, corr, p in pearson_summary:
        if p < adjusted_alpha:
            significant_vars.append((col, corr))

    if significant_vars:
        best_var, best_corr = None, None
        max_corr = -1

        for col, corr in significant_vars:
            if abs(corr) > max_corr:
                best_var = col
                best_corr = corr
                max_corr = abs(corr)

        logger.info(f"Strongest correlated variable: {best_var} (r={best_corr:.3f})")
    else:
        logger.info("No variables remain significant after correction.")

    return {
        "total_number_of_countries": total_number_of_countries,
        "total_years": total_years,
        "top_3_regions_by_mean_happiness_score": top_3_regions_by_mean_happiness_score,
        "bottom_3_regions_by_mean_happiness_score": bottom_3_regions_by_mean_happiness_score,
        "t_test_2019_2020": t_test_result,
        "most_correlated_variable": best_var if significant_vars else None
    }

@flow
def data_pipeline():

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 1: Load Multiple Years of Data")
    files = path_csv()
    df_main = read_csv_file(files)
    save_csv(df_main)
    log_columns(df_main)
    db_clean = clean_and_transform(df_main)

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 2: Descriptive Statistics")
    happiness_score_descr_stat(db_clean)

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 3: Visual Exploration")
    plot_hist(db_clean)
    plot_box(db_clean)
    gdp_vs_happ(db_clean)
    corr_heatmap(db_clean)

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 4: Hypothesis Testing")
    t_test_result = stats_test(db_clean)
    stats_second_test(db_clean)

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 5: Correlation and Multiple Comparisons")
    pearson_summary, adjusted_alpha = value_pearson(db_clean)

    logger = get_run_logger()
    logger.info("\n")
    logger.info("Task 6: Summary Report")
    summary_data = summary_report(db_clean, adjusted_alpha, t_test_result, pearson_summary)

    return summary_data

if __name__ == "__main__":
    data_pipeline()