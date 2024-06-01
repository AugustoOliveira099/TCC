import pandas as pd
import logging
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main() -> None:
    logging.info('Read data')
    datafile_path = '../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)

    # Create embedding matrix
    logging.info('Create embedding matrix (this may take about 5 minutes)')
    embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(embedding.values)

    # Normalize matrix
    logging.info('Normalize matrix')
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # KMeans without t-SNE
    logging.info('Init KMeans without t-SNE')
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(matrix_scaled)
    labels = kmeans.labels_
    df['cluster_without_tsne'] = labels

    # Apply t-SNE to reduce dimensionality
    logging.info('Apply t-SNE')
    tsne_new = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=200)
    matrix_tsne = tsne_new.fit_transform(matrix_scaled)

    # KMeans with t-SNE
    logging.info('Init KMeans with t-SNE')
    kmeans_tsne = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans_tsne.fit(matrix_tsne)
    labels_tsne = kmeans_tsne.labels_
    df['cluster_with_tsne'] = labels_tsne

    # Save the results to a CSV file
    logging.info('Save data with clusters')
    df.drop(['title', 'content', 'combined', 'n_tokens', 'target'], axis=1, inplace=True)
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
