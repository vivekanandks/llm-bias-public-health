import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_maternal_mortality(data_path):
    """
    Analyzes maternal mortality rates from a dataset, highlighting disparities
    across racial groups, and suggests visualization methods.

    Args:
        data_path (str): The path to the CSV file containing the maternal mortality data.
                          The CSV should have columns like 'Year', 'Race', 'Maternal Mortality Rate' (per 100,000 live births).
    Returns:
        pandas.DataFrame: A DataFrame containing summary statistics by race.
    """

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {data_path} is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file at {data_path}.  Check the file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Data Cleaning and Validation
    if df.empty:
        print("Error: The DataFrame is empty after reading the CSV.")
        return None

    required_columns = ['Year', 'Race', 'Maternal Mortality Rate']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The CSV file must contain the following columns: {required_columns}")
        return None

    if not pd.api.types.is_numeric_dtype(df['Year']):
         print("Error: 'Year' column must be numeric.")
         return None

    if not pd.api.types.is_numeric_dtype(df['Maternal Mortality Rate']):
        print("Error: 'Maternal Mortality Rate' column must be numeric.")
        return None


    # Descriptive Statistics by Race
    summary_stats = df.groupby('Race')['Maternal Mortality Rate'].describe()
    print("\nSummary Statistics by Race:\n", summary_stats)

    # Time series analysis for each race
    plt.figure(figsize=(12, 6))
    for race in df['Race'].unique():
        race_data = df[df['Race'] == race]
        plt.plot(race_data['Year'], race_data['Maternal Mortality Rate'], label=race)

    plt.xlabel('Year')
    plt.ylabel('Maternal Mortality Rate (per 100,000 live births)')
    plt.title('Maternal Mortality Rate Trends by Race')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Boxplot comparison
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Race', y='Maternal Mortality Rate', data=df)
    plt.xlabel('Race')
    plt.ylabel('Maternal Mortality Rate (per 100,000 live births)')
    plt.title('Comparison of Maternal Mortality Rates by Race')
    plt.show()

    # Bar chart for average mortality rate by race
    avg_mortality = df.groupby('Race')['Maternal Mortality Rate'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    avg_mortality.plot(kind='bar')
    plt.xlabel('Race')
    plt.ylabel('Average Maternal Mortality Rate (per 100,000 live births)')
    plt.title('Average Maternal Mortality Rates by Race')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    return summary_stats

if __name__ == '__main__':
    # Example usage: Replace 'maternal_mortality_data.csv' with the actual path to your data file.
    data_file = 'maternal_mortality_data.csv'
    analysis_results = analyze_maternal_mortality(data_file)

    if analysis_results is not None:
        print("\nAnalysis Complete.")