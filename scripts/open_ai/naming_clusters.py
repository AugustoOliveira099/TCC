import os
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# create OpenAi client
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

# Clusters number
n_clusters = 4

# Reading a news article which belongs to each group.
news_per_cluster = 5

logging.info("Read data")
cluters_df = pd.read_csv('../../data/noticias_ufrn_clusters.csv')

def names_clusters(cluster_column):
    for i in range(n_clusters):
        print(f"Cluster {i} Tema:", end=" ")

        reviews = "\n\n\n".join(
            cluters_df[cluters_df[cluster_column] == i]
            .combined.str.replace("Título: ", "")
            .str.replace("; Conteúdo: ", ":  ")
            .sample(news_per_cluster, random_state=43)
            .values
        )

        messages = [
            {"role": "user", "content": f'Entre os temas "Informes", "Ciências", "Eventos" e "Vagas", qual deles as notícias abaixo tem em comum?\n\nNotícias:\n"""\n{reviews}\n"""\n\nTema:'}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        print(response.choices[0].message.content.replace("\n", ""))

        sample_cluster_rows = cluters_df[cluters_df[cluster_column] == i].sample(news_per_cluster, random_state=43)
        for j in range(news_per_cluster):
            print(sample_cluster_rows.combined.str[:120].values[j])

        print("-" * 150)

def main() -> None:
    logging.info("Naming clusters with t-SNE\n")
    names_clusters("cluster_with_tsne")

    print ("\n\n\n")
    print("#" * 150)
    print ("\n\n\n")

    logging.info("Naming clusters without t-SNE\n")
    names_clusters("cluster_without_tsne")

main()
