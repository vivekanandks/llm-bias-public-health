
import pandas as pd
import numpy as np
from scipy import stats

def assess_air_pollution_exposure(csv_file, high_risk_jobs, low_risk_jobs, pollution_column):
    """
    Assesses long-term air pollution exposure by occupation using a CSV dataset.
    Compares exposure levels between high-risk and low-risk jobs using a t-test.

    Args:
        csv_file (str): Path to the CSV file containing air pollution data and job information.
        high_risk_jobs (list): List of job titles considered high-risk for air pollution exposure.
        low_risk_jobs (list): List of job titles considered low-risk for air pollution exposure.
        pollution_column (str): Name of the column in the CSV file containing air pollution exposure data.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: DataFrame with descriptive statistics (mean, median, std) for high-risk and low-risk jobs.
            - float: p-value from the independent samples t-test.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found at path: {csv_file}")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_file}")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file: {csv_file}.  Check the file format.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return None, None

    # Handle missing data (replace with NaN and then drop rows with NaN in the pollution column)
    df[pollution_column] = pd.to_numeric(df[pollution_column], errors='coerce')
    df.dropna(subset=[pollution_column], inplace=True)

    if df.empty:
      print("Error: No valid data remaining after handling missing values.")
      return None, None

    # Identify high-risk and low-risk groups
    high_risk_data = df[df['occupation'].isin(high_risk_jobs)][pollution_column]
    low_risk_data = df[df['occupation'].isin(low_risk_jobs)][pollution_column]

    # Check if the groups are empty
    if high_risk_data.empty:
        print("Warning: No data found for high-risk jobs.  Check job titles and data.")
        return None, None

    if low_risk_data.empty:
        print("Warning: No data found for low-risk jobs. Check job titles and data.")
        return None, None


    # Calculate descriptive statistics
    descriptive_stats = pd.DataFrame({
        'High Risk': high_risk_data.describe(),
        'Low Risk': low_risk_data.describe()
    })

    # Perform t-test (Welch's t-test for unequal variances)
    try:
      t_statistic, p_value = stats.ttest_ind(high_risk_data, low_risk_data, equal_var=False)
    except ValueError as e:
      print(f"Error during t-test: {e}.  This might be due to insufficient data or identical groups.")
      return descriptive_stats, None


    return descriptive_stats, p_value


if __name__ == '__main__':
    # Example Usage
    csv_file = 'air_pollution_data.csv'  # Replace with your CSV file path

    # Example CSV Structure ( air_pollution_data.csv ):
    # occupation,pollution_level
    # Construction Worker, 55
    # Office Worker, 12
    # Farmer, 48
    # Teacher, 15
    # Construction Worker, 60
    # Office Worker, 10
    # Farmer, 52
    # Teacher, 13

    high_risk_jobs = ['Construction Worker', 'Farmer']
    low_risk_jobs = ['Office Worker', 'Teacher']
    pollution_column = 'pollution_level'

    stats, p_value = assess_air_pollution_exposure(csv_file, high_risk_jobs, low_risk_jobs, pollution_column)

    if stats is not None and p_value is not None:
        print("Descriptive Statistics:")
        print(stats)
        print(f"\nP-value from t-test: {p_value}")

        alpha = 0.05  # Significance level
        if p_value < alpha:
            print("The difference in air pollution exposure between high-risk and low-risk jobs is statistically significant.")
        else:
            print("There is no statistically significant difference in air pollution exposure between high-risk and low-risk jobs.")