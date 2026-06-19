python version 3.11

Sergei Patrushev

# World Happiness Analytics Pipeline

An end-to-end data analytics pipeline built with Prefect for automated processing and analysis of the World Happiness Report dataset across multiple years.

The pipeline performs data ingestion, validation, cleaning, transformation, statistical analysis, hypothesis testing, correlation analysis, and visualization generation. It combines ETL engineering practices with exploratory and inferential statistics to produce reproducible insights into global happiness trends.

## Features

* Automated ingestion and merging of multi-year CSV datasets
* Data cleaning and preprocessing
* Missing value handling and data quality checks
* Workflow orchestration with Prefect
* Structured logging and retry mechanisms
* Descriptive statistical analysis
* Welch's t-tests for hypothesis testing
* Pearson correlation analysis
* Bonferroni correction for multiple comparisons
* Automated visualization generation
* Summary report generation

## Technologies

* Python
* Prefect
* Pandas
* NumPy
* SciPy
* Matplotlib
* Seaborn

## Statistical Methods

* Descriptive Statistics
* Exploratory Data Analysis (EDA)
* Welch's Independent Samples t-test
* Pearson Correlation Coefficient
* Multiple Testing Correction (Bonferroni)

## Generated Outputs

* Merged dataset
* Happiness score distribution histogram
* Year-over-year happiness boxplots
* GDP vs Happiness scatter plot
* Correlation heatmap
* Statistical summary report

## Skills

Data Engineering • ETL Development • Workflow Orchestration • Data Cleaning • Statistical Analysis • Hypothesis Testing • Correlation Analysis • Data Visualization • Analytics Automation • Reproducible Data Pipelines


additional
--------------------------------------------------------------------------------------
Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of 
the linear relationship between two variables. Its values range from −1 to +1, where:

+1 indicates a perfect positive linear relationship,
−1 indicates a perfect negative linear relationship,
0 indicates no linear relationship.

In simple terms, it measures how closely the points in a scatter plot follow a straight line.

import numpy as np
r = np.corrcoef(x, y)[0, 1]
---------------------------------------------------------------------------------------