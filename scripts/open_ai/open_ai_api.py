import logging
import pandas as pd
import sys
import os

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.eda import plot_frequency_of_tokens, \
                       calc_number_tokens

from utils.data_collection import fetch_embedding

from utils.preprocessing import remove_html, \
                                 combine_columns, \
                                 remove_long_news

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main() -> None:
    # embedding model parameters
    embedding_model = "text-embedding-3-large"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-3-large
    max_tokens = 8191  # the maximum input for text-embedding-3-large is 8191

    # Lê arquivos csv com os dados das notícias
    logging.info('Read news')
    df = pd.read_csv('../../data/news.csv')

    # Combina colunas do dataframe
    logging.info('Combine columns')
    df_combined = combine_columns(df, 'title', 'content', 'combined')

    # Remove HTML do texto
    df_combined['combined'] = df_combined['combined'].apply(remove_html)

    logging.info('Calculate the number of tokens for each cell in "combined" column')
    df_tokens = calc_number_tokens(df_combined, 'combined', 'n_tokens', embedding_encoding)

    logging.info('Remove news that are too long to embed')
    df_tokens = remove_long_news(df_tokens, 'n_tokens', max_tokens)

    # Plota frequência de tokens no texto
    logging.info('Plot the token frequency')
    fig_path = '../../images/token_frequency.png'
    plot_frequency_of_tokens(df_tokens, fig_path)

    # Calculate the price of the OpenAI embedding
    cost = 0.00013 # Cost per 1000 tokens to the text-embedding-3-large model
    num_tokens = df_tokens.n_tokens.sum()
    print(f"There are {num_tokens} tokens. Estimated priced is U$ {(num_tokens/1000*cost)}.")

    # # Embed combined news
    # logging.info('Embed combined news (this may take about 3 hours)')
    # df_tokens["embedding"] = df_tokens['combined'].apply(lambda x: fetch_embedding(x, embedding_model))

    # # Export the output into a csv file
    # df_tokens.to_csv("../../data/noticias_ufrn_embeddings.csv", index=False)

main()