import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import umap

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
        project="k-means-with-umap",
        config={
            'n_clusters': n_clusters,
            'init': 'random',
            'n_init': 40,
            'max_iter': 500,
            'tol': 1,
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

    # Apply UMAP to reduce dimensionality
    logging.info('Apply UMAP')
    umap_model = umap.UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
    )
    vis_dims_new = umap_model.fit_transform(matrix_scaled)

    # Save UMAP
    umap_path = "umap.pkl"
    joblib.dump(umap_model, umap_path)
    umap_artifact = wandb.Artifact(name="umap", type="model")
    umap_artifact.add_file(umap_path, name="umap.pkl")
    wandb.log_artifact(umap_artifact)

    # KMeans with UMAP
    logging.info('Init KMeans with UMAP')
    kmeans_umap = KMeans(
        n_clusters=config.n_clusters,
        init=config.init,
        n_init=config.n_init,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state
    )
    clusters_umap = kmeans_umap.fit_predict(vis_dims_new)
    df['cluster_with_umap'] = clusters_umap

    # Calculando a Silhouette Score
    silhouette_umap = silhouette_score(vis_dims_new, clusters_umap, sample_size=2000)

    logging.info(f"Silhouette Score with UMAP: {silhouette_umap}")

    # Save KMeans model in Weights and Biases
    logging.info('Saving KMeans model')
    joblib.dump(kmeans_umap, 'kmeans_umap.pkl')
    run.link_model(path="./kmeans_umap.pkl", registered_model_name="kmeans_umap")

    # Save the results into CSV file
    logging.info('Save data with clusters')
    df.drop(['title', 'content', 'n_tokens', 'target'], axis=1, inplace=True)
    umap_df = pd.DataFrame(vis_dims_new, columns=['umap1', 'umap2'])
    df = df.reset_index(drop=True)  # Reinicia os índices do DataFrame original
    df = pd.concat([df, umap_df], axis=1)
    df.to_csv('../../data/kmeans/noticias_umap.csv', index=False)

    # Visualize the KMeans clusters with UMAP
    plt.figure(figsize=(12, 6))
    x = [x for x, y in vis_dims_new]
    y = [y for x, y in vis_dims_new]
    for category, color in enumerate(['purple', 'green', 'red', 'blue']):
        xs = np.array(x)[df.cluster_with_umap == category]
        ys = np.array(y)[df.cluster_with_umap == category]
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
    plt.title('KMeans Clusters with UMAP')

    # Save image locally
    image_path = '../../images/clusters_with_umap.png'
    plt.savefig(image_path)

    # Save image into Weights and Biases
    wandb.log({"Clusters with UMAP": wandb.Image(image_path)})

    # # Stop the carbon tracker
    # emissions = tracker.stop()
    # logging.info(f'carbon_emissions: {emissions}')

    # # Log carbon emissons artifact
    # artifact = wandb.Artifact(name="carbon_emissions", type="dataset")
    # artifact.add_file('emissions.csv', name='emissions.csv')
    # wandb.log_artifact(artifact)

    run.finish()

main()
