
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_vaccination_rates(data_path):
    """
    Visualizes country-level vaccination rates from a dataset, highlighting rural
    versus urban coverage disparities.

    Args:
        data_path (str): Path to the CSV file containing vaccination data.
                          The CSV should have columns: 'Country', 'Rural_Coverage',
                          'Urban_Coverage'.
    """

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {data_path} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file at {data_path}. Check the format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Data Cleaning and Validation
    if not {'Country', 'Rural_Coverage', 'Urban_Coverage'}.issubset(df.columns):
        print("Error: The CSV file must contain 'Country', 'Rural_Coverage', and 'Urban_Coverage' columns.")
        return

    if df.isnull().any().any():
        print("Warning: Missing values found. Imputing with the mean.")
        df = df.fillna(df.mean(numeric_only=True))

    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in ['Rural_Coverage', 'Urban_Coverage']):
        print("Error: 'Rural_Coverage' and 'Urban_Coverage' columns must be numeric.")
        return

    # Ensure coverage rates are within 0-100 range
    for col in ['Rural_Coverage', 'Urban_Coverage']:
        if (df[col] < 0).any() or (df[col] > 100).any():
            print(f"Warning: {col} contains values outside the 0-100 range. Clipping to 0-100.")
            df[col] = df[col].clip(0, 100)

    # Data Visualization
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Sort by difference for visual emphasis
    df['Coverage_Difference'] = abs(df['Rural_Coverage'] - df['Urban_Coverage'])
    df = df.sort_values('Coverage_Difference', ascending=False)
    df = df.drop('Coverage_Difference', axis=1)

    # Melt the dataframe for easier plotting
    df_melted = pd.melt(df, id_vars=['Country'], value_vars=['Rural_Coverage', 'Urban_Coverage'],
                          var_name='Coverage_Type', value_name='Coverage_Rate')

    # Create the bar plot
    ax = sns.barplot(x='Country', y='Coverage_Rate', hue='Coverage_Type', data=df_melted, palette="viridis")

    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.title('Country-Level Vaccination Coverage: Rural vs. Urban', fontsize=16)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Vaccination Coverage (%)', fontsize=12)
    plt.legend(title='Coverage Type')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage with a dummy CSV file.  Replace 'vaccination_data.csv'
    # with the actual path to your data.
    # You'll need to create a sample CSV for testing.
    # Example vaccination_data.csv content:
    # Country,Rural_Coverage,Urban_Coverage
    # USA,65,80
    # Canada,75,85
    # Mexico,50,60
    # Brazil,40,70
    # Argentina,60,75

    dummy_data = {'Country': ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina'],
                  'Rural_Coverage': [65, 75, 50, 40, 60],
                  'Urban_Coverage': [80, 85, 60, 70, 75]}
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('vaccination_data.csv', index=False)

    visualize_vaccination_rates('vaccination_data.csv')