import tiktoken
import pandas as pd
import logging
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import openai

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('-------- STARTING DATA PREPROCESSING --------')

# Lê arquivos csv com os dados
logging.info('Read data')
df1 = pd.read_csv('data/first_part.csv')
df2 = pd.read_csv('data/second_part.csv')

# Faz o merge dos dataframes
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

# Combina os dados da coluna tile com os dados da coluna content
logging.info('Combine data')
df_merged['combined'] = 'Título: ' + df_merged['title'].str.strip() + '; Conteúdo: ' + df_merged['content'].str.strip()

# Função para remover HTML de uma string
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Remove HTML do texto
logging.info('Remove HTML from content')
df_merged['combined'] = df_merged['combined'].apply(lambda x: remove_html(x))

# Salva dataframe como arquivo csv
logging.info('Save dataframe as csv file')
df_merged.to_csv('data/merged_news.csv', index=False)

# embedding model parameters
embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-3-large
max_tokens = 8191  # the maximum input for text-embedding-3-large is 8191

# Calculate the number of tokens
logging.info('Calculate the number of tokens for each cell in column "combined"')
encoding = tiktoken.get_encoding(embedding_encoding)
df_merged["n_tokens"] = df_merged.combined.apply(lambda x: len(encoding.encode(x)))

logging.info('Plot the token frequency')
# Set the style of the plot
plt.style.use('ggplot')

# Create the histogram using the 'Token' column
plt.hist(df_merged['n_tokens'], bins=30, edgecolor='white')

# Add labels for x and y axes
plt.xlabel('Token Value')
plt.ylabel('Frequency')

# Calculate the mean of the 'n_tokens' column
mean_value = df_merged['n_tokens'].mean()

# Add a vertical line for the mean value
plt.axvline(mean_value, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')

# Find the maximum value of the x-axis
max_x_value = df_merged['n_tokens'].max()

# Add an arrow pointing to the maximum x value
plt.annotate(f'Max: {max_x_value}', xy=(max_x_value, 0), xycoords='data', xytext=(max_x_value - 30, 25),
             textcoords='data', arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10)

# Add the legend
plt.legend()

# Save the plot as an image file (e.g., 'histogram.png')
plt.savefig('histogram.png')

# Close the plot
plt.close()

# Calculate the price of the OpenAI embedding
print(f"Estimated priced is U$ {(df_merged.n_tokens.sum()/1000*0.00013)}.")