import logging
import pandas as pd
import sys
import os

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.append(relative_path)

from utils import plot_frequency_of_tokens, \
                  remove_html_from_df, \
                  combine_columns, \
                  calc_number_tokens, \
                  remove_long_news

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# embedding model parameters
embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-3-large
max_tokens = 8191  # the maximum input for text-embedding-3-large is 8191

logging.info('-------- STARTING DATA PREPROCESSING --------')

# Lê arquivos csv com os dados das notícias
logging.info('Read data')
df1 = pd.read_csv('data/first_part.csv')
df2 = pd.read_csv('data/second_part.csv')

# Faz o merge dos dataframes
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

# Combina colunas do dataframe
df_combined = combine_columns(df_merged, 'title', 'content', 'combined')

# Remove HTML do texto
df_combined = remove_html_from_df(df_combined, 'combined')

logging.info('Calculate the number of tokens for each cell in "combined" column')
df_tokens = calc_number_tokens(df_combined, 'combined', 'n_tokens', embedding_encoding)

logging.info('Remove news that are too long to embed')
df_tokens = remove_long_news(df_tokens, 'n_tokens', max_tokens)

# Plota frequência de tokens no texto
plot_frequency_of_tokens(df_tokens)

# Calculate the price of the OpenAI embedding
cost = 0.00013 # Cost per 1000 tokens to the text-embedding-3-large model
print(f"Estimated priced is U$ {(df_tokens.n_tokens.sum()/1000*cost)}.")

# # Embed combined news
# logging.info('Embed combined news (this may take about 3 hours)')
# df_tokens["embedding"] = df_tokens['combined'].apply(lambda x: get_embedding(x, embedding_model))

# # Export the output into a .csv file
# df_tokens.to_csv("data/noticias_ufrn_embeddings.csv", index=False)
