"""
Script com funções que serão utilizadas 
"""

# Importa bibliotecas
import pandas as pd
import logging
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# create OpenAi client
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def plot_frequency_of_tokens(df: pd.DataFrame) -> None:
    logging.info('Plot the token frequency')
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

    # Save the plot as an image file (e.g., 'histogram.png')
    plt.savefig('histogram.png')

    # Close the plot
    plt.close()

# Função para remover HTML de uma string
def remove_html(text) -> str:
    soup = BeautifulSoup(text, "html.parser")
    # Remover todas as tags 'figcaption' e seu conteúdo
    for figcaption in soup.find_all('figcaption'):
        figcaption.decompose()
    # Remover linhas em branco e retornar o texto sem HTML
    return '\n'.join([line for line in soup.get_text().split('\n') if line.strip()])

# Remove HTML do texto
def remove_html_from_df(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    logging.info(f'Remove HTML from column {target_column}')
    df[target_column] = df[target_column].apply(lambda x: remove_html(x))
    return df

# Combina colunas de um dataframe
def combine_columns(df: pd.DataFrame,
                    title_column: str,
                    content_column: str,
                    result_column: str = 'combined') -> pd.DataFrame:
    # Combina os dados da coluna tile com os dados da coluna content
    logging.info('Combine data')
    df[result_column] = 'Título: ' + df[title_column].str.strip() + '; Conteúdo: ' + df[content_column].str.strip()
    return df

# Calculate the number of tokens
def calc_number_tokens(df: pd.DataFrame,
                       target_column: str,
                       result_column: str,
                       embedding_encoding: str) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(embedding_encoding)
    df[result_column] = df[target_column].apply(lambda x: len(encoding.encode(x)))
    return df

# Remove news that are too long to embed
def remove_long_news(df: pd.DataFrame, column_tokens: str, max_tokens: int) -> pd.DataFrame:
    # Remove news that are too long to embed
    sum_news_too_long = (df[column_tokens] > max_tokens).sum()
    logging.info(f'{sum_news_too_long} news are too long to embed')
    df = df[df[column_tokens] <= max_tokens]
    return df

# Requisita embedding à API da OpenAi
def get_embedding(text: str, model="text-embedding-3-large") -> list[float]:
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
