import pandas as pd
import matplotlib.pyplot as plt
import tiktoken

def plot_frequency_of_tokens(df: pd.DataFrame, fig_path: str) -> None:
    """
    Plots and saves news token frequency graph.

    Args:
        df (pd.DataFrame): dataframe with the data to be plotted.
        fig_path (str): Relative path to save the figure.
    
    Returns:
        None
    """
    # Set the style of the plot
    plt.style.use('ggplot')

    # Create the histogram using the 'Token' column
    plt.hist(df['n_tokens'], bins=30, edgecolor='white')

    # Add labels for x and y axes
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')

    # Calculate the mean of the 'n_tokens' column
    mean_value = df['n_tokens'].mean()

    # Add a vertical line for the mean value
    plt.axvline(mean_value, color='blue',
                linestyle='dashed',
                linewidth=1,
                label=f'Mean: {mean_value:.2f}')

    # Find the maximum value of the x-axis
    max_x_value = df['n_tokens'].max()

    # Add an arrow pointing to the maximum x value
    plt.annotate(f'Max: {max_x_value}',
                 xy=(max_x_value, 0),
                 xycoords='data',
                 xytext=(max_x_value - 30, 25),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->', lw=1.5),
                 fontsize=10)

    # Add the legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig(fig_path)

    # Close the plot
    plt.close()

def calc_number_tokens(df: pd.DataFrame,
                       target_column: str,
                       result_column: str,
                       embedding_encoding: str) -> pd.DataFrame:
    """
    Calculates the number of tokens for each entry in the target column using the specified 
        encoding and stores the result in a new column.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        target_column (str): The name of the column containing the text to be tokenized.
        result_column (str): The name of the column where the token counts will be stored.
        embedding_encoding (str): The encoding to use for tokenization.

    Returns:
        pd.DataFrame: The DataFrame with an additional column containing the token counts.
    """
    encoding = tiktoken.get_encoding(embedding_encoding)
    df[result_column] = df[target_column].apply(lambda x: len(encoding.encode(x)))
    return df
