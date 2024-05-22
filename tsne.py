import pandas as pd
import logging
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('Read data')
datafile_path = 'data/noticias_ufrn_embeddings.csv'
df = pd.read_csv(datafile_path)

# Create embedding matrix
logging.info('Create embedding matrix (this may take about 10 minutes)')
initial_time = time.time()
df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embedding.values)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to create embedding matrix was {total_time} seconds.')

# Normalize matrix
logging.info('Normalize matrix')
initial_time = time.time()
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to normalize matrix was {total_time} seconds.')

# KMeans without t-SNE
logging.info('Init KMeans without t-SNE')
initial_time = time.time()
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
kmeans.fit(matrix_scaled)
labels = kmeans.labels_
df['cluster_without_tsne'] = labels
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to KMeans without t-SNE was {total_time} seconds.')

# Apply t-SNE to reduce dimensionality
logging.info('Apply t-SNE')
initial_time = time.time()
tsne_new = TSNE(n_components=2, perplexity=50, random_state=42, learning_rate=200)
matrix_tsne = tsne_new.fit_transform(matrix_scaled)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to apply t-SNE was {total_time} seconds.')

# KMeans with t-SNE
logging.info('Init KMeans with t-SNE')
initial_time = time.time()
kmeans_tsne = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
kmeans_tsne.fit(matrix_tsne)
labels_tsne = kmeans_tsne.labels_
df['cluster_with_tsne'] = labels_tsne
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time for KMeans with t-SNE was {total_time} seconds.')

# Save the results to a CSV file
logging.info('Save data with clusters')
df.to_csv('data/noticias_ufrn_clusters.csv', index=False)

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
plt.savefig('images/clusters_with_tse.png')
