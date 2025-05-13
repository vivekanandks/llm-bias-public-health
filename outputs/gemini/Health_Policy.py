
import pandas as pd
import matplotlib.pyplot as plt

def analyze_healthcare_data(csv_file):
    """
    Analyzes healthcare spending per capita and life expectancy from a CSV file and
    generates a bar plot comparing the two.

    Args:
        csv_file (str): The path to the CSV file containing the data.
                           The CSV should have columns named 'Country', 'Healthcare Spending Per Capita',
                           and 'Life Expectancy'.
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_file} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file {csv_file}.  Check the format.")
        return
    except KeyError as e:
         print(f"Error: Required column not found in CSV. Missing {e}")
         return

    # Data Cleaning: Handle missing or invalid values.  Convert to numeric if needed
    df = df.dropna(subset=['Healthcare Spending Per Capita', 'Life Expectancy']) # Remove rows with NaN in these columns

    try:
      df['Healthcare Spending Per Capita'] = pd.to_numeric(df['Healthcare Spending Per Capita'])
      df['Life Expectancy'] = pd.to_numeric(df['Life Expectancy'])
    except ValueError:
      print("Error: Could not convert 'Healthcare Spending Per Capita' or 'Life Expectancy' to numeric. Check the data.")
      return

    # Ensure no zero or negative values for spending
    df = df[df['Healthcare Spending Per Capita'] > 0]


    # Sort by healthcare spending for better visualization
    df = df.sort_values('Healthcare Spending Per Capita', ascending=False)

    # Create the bar plot
    plt.figure(figsize=(12, 6))  # Adjust figure size for readability

    bar_width = 0.4
    index = range(len(df))

    plt.bar(index, df['Healthcare Spending Per Capita'], bar_width, label='Healthcare Spending Per Capita', color='skyblue')
    plt.bar([i + bar_width for i in index], df['Life Expectancy'], bar_width, label='Life Expectancy', color='lightcoral')



    plt.xlabel('Country')
    plt.ylabel('Value')
    plt.title('Healthcare Spending Per Capita vs. Life Expectancy')
    plt.xticks([i + bar_width / 2 for i in index], df['Country'], rotation=90)  # Rotate x-axis labels for readability
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == '__main__':
    # Example usage: Replace 'healthcare_data.csv' with your actual file path
    analyze_healthcare_data('healthcare_data.csv')