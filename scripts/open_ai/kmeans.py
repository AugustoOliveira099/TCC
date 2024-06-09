import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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
        project="k-means-model",
        config={
            'n_clusters': n_clusters,
            'init': 'k-means++',
            'n_init': 20,
            'max_iter': 500,
            'tol': 0.00001,
            'random_state': 42,
        }
    )
    config = wandb.config

    logging.info('Read data')
    datafile_path = '../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)

    # Create embedding matrix
    logging.info('Create embedding matrix (this may take about 5 minutes)')
    embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(embedding.values)

    # Normalize matrix
    logging.info('Normalize matrix')
    scaler = MinMaxScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # KMeans without t-SNE
    logging.info('Init KMeans without t-SNE')
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        init=config.init,
        n_init=config.n_init,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state
    )
    kmeans.fit(matrix_scaled)
    labels = kmeans.labels_
    df['cluster_without_tsne'] = labels

    # Save KMeans model in Weight and Biases
    logging.info('Saving KMeans model')
    joblib.dump(kmeans, 'kmeans.pkl')
    run.link_model(path="./kmeans.pkl", registered_model_name="kmeans") # Log and link the model to the Model Registry

    # Apply t-SNE to reduce dimensionality
    logging.info('Apply t-SNE')
    tsne_new = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=200)
    matrix_tsne = tsne_new.fit_transform(matrix_scaled)

    # KMeans with t-SNE
    logging.info('Init KMeans with t-SNE')
    kmeans_tsne = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, max_iter=500, tol=0.00001, random_state=42)
    kmeans_tsne.fit(matrix_tsne)
    labels_tsne = kmeans_tsne.labels_
    df['cluster_with_tsne'] = labels_tsne

    # Save the results into CSV file
    logging.info('Save data with clusters')
    df.drop(['title', 'content', 'n_tokens', 'target'], axis=1, inplace=True)
    tsne_df = pd.DataFrame(matrix_tsne, columns=['tsne1', 'tsne2'])
    df = df.reset_index(drop=True)  # Reinicia os índices do DataFrame original
    df = pd.concat([df, tsne_df], axis=1)
    df.to_csv('../../data/noticias_ufrn_clusters.csv', index=False)

    # Visualize the KMeans clusters with t-SNE
    plt.figure(figsize=(12, 6))
    x = [x for x, y in matrix_tsne]
    y = [y for x, y in matrix_tsne]
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
    plt.savefig('../../images/clusters_with_tse.png')

    run.finish()

main()
