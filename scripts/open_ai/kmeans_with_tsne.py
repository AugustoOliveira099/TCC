import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import matplotlib.pyplot as plt
import openTSNE
from codecarbon import EmissionsTracker
from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from dotenv import load_dotenv

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
        project="k-means-with-tsne",
        config={
            'n_clusters': n_clusters,
            'init': 'random',
            'n_init': 40,
            'max_iter': 500,
            'tol': 0.001,
            'random_state': 42,
            'n_components': 2,
            'n_neighbors': 20,
            'min_dist': 0.1,
            'metric': 'euclidean',
        }
    )
    config = wandb.config

    # # Inicialize o rastreador de emissões de carbono
    # tracker = EmissionsTracker()
    # tracker.start()

    logging.info('Read data')
    datafile_path = '../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)
    classified_news = pd.read_csv('../../data/classified_news.csv')

    # Exclui as notícias já classificadas manualmente
    df = df[~df['title'].isin(classified_news['title'])]

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

    # Apply t-SNE to reduce dimensionality
    logging.info('Apply t-SNE')
    tsne = openTSNE.TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        learning_rate=200
    )
    vis_dims_new = tsne.fit(matrix_scaled)

    # Save t-SNE
    tsne_path = "tsne.pkl"
    joblib.dump(vis_dims_new, tsne_path)
    tsne_artifact = wandb.Artifact(name="tsne", type="model")
    tsne_artifact.add_file(tsne_path, name="tsne.pkl")
    wandb.log_artifact(tsne_artifact)

    # KMeans with T-SNE
    logging.info('Init KMeans with T-SNE')
    kmeans_tsne = KMeans(
        n_clusters=config.n_clusters,
        init=config.init,
        n_init=config.n_init,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state
    )
    clusters_tsne = kmeans_tsne.fit_predict(vis_dims_new)
    df['cluster_with_tsne'] = clusters_tsne

    # Calculando a Silhouette Score
    silhouette_tsne = silhouette_score(vis_dims_new, clusters_tsne, sample_size=2000)

    logging.info(f"Silhouette Score with T-SNE: {silhouette_tsne}")

    # Save KMeans model in Weights and Biases
    logging.info('Saving KMeans model')
    kmeans_tsne_path = './kmeans_tsne.pkl'
    joblib.dump(kmeans_tsne, kmeans_tsne_path)
    logged_artifact = run.log_artifact(
        kmeans_tsne_path,
        name="kmeans_tsne",
        type="model"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-model/kmeans-tsne"
    )

    # Save the results into CSV file
    logging.info('Save data with clusters')
    df.drop(['title', 'content', 'n_tokens', 'target'], axis=1, inplace=True)
    tsne_df = pd.DataFrame(vis_dims_new, columns=['tsne1', 'tsne2'])
    df = df.reset_index(drop=True)  # Reinicia os índices do DataFrame original
    df = pd.concat([df, tsne_df], axis=1)
    df.to_csv('../../data/kmeans/noticias_tsne.csv', index=False)

    # Visualize the KMeans clusters with t-SNE
    plt.figure(figsize=(12, 6))
    x = [x for x, y in vis_dims_new]
    y = [y for x, y in vis_dims_new]
    for category, color in enumerate(['purple', 'green', 'red', 'blue']):
        xs = np.array(x)[df.cluster_with_tsne == category]
        ys = np.array(y)[df.cluster_with_tsne == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)
        avg_x = xs.mean()
        avg_y = ys.mean()

        if color == 'purple':
            marker_color = 'yellow'
        elif color == 'green':
            marker_color = 'orange'
        elif color == 'red':
            marker_color = 'cyan'
        else:
            marker_color = 'magenta'

        plt.scatter(avg_x, avg_y, marker='x', color=marker_color, s=100)
    plt.title('KMeans Clusters with t-SNE')

    # Save image locally
    image_path = '../../images/clusters_with_tsne.png'
    plt.savefig(image_path)

    # Save image into Weights and Biases
    wandb.log({"Clusters with T-SNE": wandb.Image(image_path)})

    # # Stop the carbon tracker
    # emissions = tracker.stop()
    # logging.info(f'carbon_emissions: {emissions}')

    # # Log carbon emissons artifact
    # artifact = wandb.Artifact(name="carbon_emissions", type="dataset")
    # artifact.add_file('emissions.csv', name='emissions.csv')
    # wandb.log_artifact(artifact)

    run.finish()

main()
