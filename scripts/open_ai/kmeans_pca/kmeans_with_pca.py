import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import matplotlib.pyplot as plt
import openTSNE
from codecarbon import track_emissions
from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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
        project="k-means-with-pca",
        config={
            'n_clusters': n_clusters,
            'init': 'random',
            'n_init': 30,
            'max_iter': 500,
            'tol': 0.1,
            'random_state': 42,
            'n_components': 2,
            'svd_solver': 'randomized',
        }
    )
    config = wandb.config

    # # Inicialize o rastreador de emissões de carbono
    # tracker = EmissionsTracker()
    # tracker.start()

    logging.info('Read data')
    datafile_path = '../../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)
    classified_news = pd.read_csv('../../../data/classified_news.csv')

    # Exclui as notícias já classificadas manualmente
    df = df[~df['title'].isin(classified_news['title'])]

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

        # Apply PCA to reduce dimensionality
        logging.info('Apply PCA')
        pca_model = PCA(
            n_components=config.n_components,
            svd_solver=config.svd_solver,
            random_state=config.random_state,
        )
        vis_dims_new = pca_model.fit_transform(matrix_scaled)

        # Save PCA
        pca_path = "pca.pkl"
        joblib.dump(pca_model, pca_path)
        pca_artifact = wandb.Artifact(name="pca", type="model")
        pca_artifact.add_file(pca_path, name="pca.pkl")
        wandb.log_artifact(pca_artifact)

        # KMeans with PCA
        logging.info('Init KMeans with PCA')
        kmeans_pca = KMeans(
            n_clusters=config.n_clusters,
            init=config.init,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tol,
            random_state=config.random_state
        )
        clusters_pca = kmeans_pca.fit_predict(vis_dims_new)
        df['cluster'] = clusters_pca

        # Calculando a Silhouette Score
        silhouette_pca = silhouette_score(vis_dims_new, clusters_pca, sample_size=2000)

        logging.info(f"Silhouette Score with PCA: {silhouette_pca}")

        # Save KMeans model into Weights and Biases
        logging.info('Saving KMeans model')
        kmeans_pca_path = './kmeans_pca.pkl'
        joblib.dump(kmeans_pca, kmeans_pca_path)
        logged_artifact = run.log_artifact(
            kmeans_pca_path,
            name="kmeans_pca",
            type="model"
        )
        run.link_artifact(
            artifact=logged_artifact,
            target_path="mlops2023-2-org/wandb-registry-model/kmeans-pca"
        )

        return vis_dims_new

    vis_dims_new = preprocessing_train_model()

    # Save emssions into Weights and Biases
    logging.info('Saving carbon emissions')
    emissions_path = './emissions.csv'
    logged_artifact = run.log_artifact(
        emissions_path,
        name="emissions_kmeans_pca",
        type="dataset"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-dataset/emissions_kmeans_pca"
    ) # Log and link the emissions to the Model Registry

    # Save the results into CSV file
    logging.info('Save data with clusters')
    dataset_path = '../../../data/kmeans/noticias_pca.csv'
    df.drop(['title', 'content', 'n_tokens', 'target'], axis=1, inplace=True)
    pca_df = pd.DataFrame(vis_dims_new, columns=['pca'])
    df = df.reset_index(drop=True)  # Reinicia os índices do DataFrame original
    df = pd.concat([df, pca_df], axis=1)
    df.to_csv(dataset_path, index=False)

    # Visualize the KMeans clusters with t-SNE
    plt.figure(figsize=(12, 6))
    x = [x for x, y in vis_dims_new]
    y = [y for x, y in vis_dims_new]
    for category, color in enumerate(['purple', 'green', 'red', 'blue']):
        xs = np.array(x)[df.cluster_with_pca == category]
        ys = np.array(y)[df.cluster_with_pca == category]
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
    image_path = '../../../images/clusters_with_pca.png'
    plt.savefig(image_path)

    # Save image into Weights and Biases
    wandb.log({"Clusters with PCA": wandb.Image(image_path)})

    # Save new dataset into Weights and Biases
    logging.info('Saving dataset into Weights and Biases')
    logged_artifact = run.log_artifact(
        dataset_path,
        name="dataset_kmeans_pca",
        type="dataset"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-dataset/dataset_kmeans_pca"
    ) # Log and link the dataset to the Model Registry

    run.finish()

main()
