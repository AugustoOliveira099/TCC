import os
import sys
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from codecarbon import track_emissions
from sklearn.metrics import silhouette_score
from codecarbon import EmissionsTracker
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from dotenv import load_dotenv

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(relative_path)

from utils.CustomKmeans import CustomKMeans

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

def main() -> None:
    # Set number of clusters
    n_clusters = 4

    # Start a run, tracking hyperparameters
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="k-means-model",
        config={
            'n_clusters': n_clusters,
            'init': 'random',
            'n_init': 30,
            'max_iter': 500,
            'tol': 0.001,
            'random_state': 3,
        }
    )
    config = wandb.config

    logging.info('Read data')
    datafile_path = '../../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)
    classified_news = pd.read_csv('../../../data/classified_news.csv')

    # Exclui as notícias já classificadas manualmente
    df = df[~df['title'].isin(classified_news['title'])]

    # Concatenando todos os textos em uma única string
    todos_os_textos = " ".join(df['combined'])

    # Gerando a nuvem de palavras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todos_os_textos)

    # Plotando a nuvem de palavras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove os eixos
    image_path = '../../../images/wordcloud_kmeans.png'
    plt.savefig(image_path)\

    @track_emissions(save_to_api=True)
    def preprocessing_train_model():
        # Create embedding matrix
        logging.info('Create embedding matrix')
        embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
        matrix = np.vstack(embedding.values)

        # Normalize matrix
        logging.info('Normalize matrix')
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)

        # Save scaler
        scaler_path = 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        artifact = wandb.Artifact(name="scaler", type="preprocessing")
        artifact.add_file(scaler_path, name="scaler.pkl")
        wandb.log_artifact(artifact)

        # KMeans
        logging.info('Init KMeans without t-SNE')
        kmeans = KMeans(
            n_clusters=config.n_clusters,
            init=config.init,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tol,
            random_state=config.random_state,
        )
        kmeans.fit(matrix_scaled)
        df['cluster'] = kmeans.labels_

        # Save KMeans model in Weights and Biases
        logging.info('Saving KMeans model')
        kmeans_path = './kmeans.pkl'
        joblib.dump(kmeans, kmeans_path)
        logged_artifact = run.log_artifact(
            kmeans_path,
            name="kmeans",
            type="model"
        )
        run.link_artifact(
            artifact=logged_artifact,
            target_path="mlops2023-2-org/wandb-registry-model/kmeans"
        ) # Log and link the model to the Model Registry

        # Calculando a Silhouette Score
        silhouette = silhouette_score(matrix_scaled, kmeans.labels_, sample_size=2000)

        wandb.log({"silhouette_score": silhouette})

    preprocessing_train_model()

    # Save emissions into Weights and Biases
    logging.info('Saving carbon emissions')
    emissions_path = './emissions.csv'
    logged_artifact = run.log_artifact(
        emissions_path,
        name="emissions_kmeans",
        type="dataset"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-dataset/emissions_kmeans"
    ) # Log and link the emissions to the Model Registry

    # Save the results into CSV file
    logging.info('Save data with clusters')
    dataset_path = '../../../data/kmeans/noticias_kmeans.csv'
    df.drop(['title', 'content', 'n_tokens', 'target'], axis=1, inplace=True)
    df.to_csv(dataset_path, index=False)

    # Save new dataset into Weights and Biases
    logging.info('Saving dataset into Weights and Biases')
    logged_artifact = run.log_artifact(
        dataset_path,
        name="dataset_kmeans",
        type="dataset"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-dataset/dataset_kmeans"
    ) # Log and link the dataset to the Model Registry

    run.finish()

main()
