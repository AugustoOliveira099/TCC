import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
from codecarbon import EmissionsTracker
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
    # Start a run, tracking hyperparameters
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="k-means-with-umap"
    )

    logging.info('Read data')
    datafile_path = '../../data/noticias_ufrn_embeddings.csv'
    df = pd.read_csv(datafile_path)
    classified_news = pd.read_csv('../../data/classified_news.csv')

    # Captura as notícias já classificadas manualmente
    df = df[df['content'].isin(classified_news['content'])]

    # Retira instancias a mais que está em classified_news
    classified_news = classified_news[classified_news['content'].isin(df['content'])]

    # Drop duplicates
    df = df.drop_duplicates(subset=['content'])
    classified_news = classified_news.drop_duplicates(subset=['content'])

    # Ordenar os DataFrames pelo conteúdo
    df = df.sort_values(by='content').reset_index(drop=True)
    classified_news = classified_news.sort_values(by='content').reset_index(drop=True)

    # Verifica se os conteúdos estão nas mesmas posições
    if df['content'].equals(classified_news['content']):
        logging.info("Os DataFrames têm o mesmo conteúdo nas mesmas posições.")
    else:
        logging.info("Os DataFrames têm conteúdo diferente ou em posições diferentes.")

    logging.info('Access and download K-Means model')
    downloaded_model_path = run.use_model('tcc-ufrn/model-registry/kmeans_umap:latest')

    # Verifica se o arquivo existe
    if os.path.exists(downloaded_model_path):
        logging.info(f"Model file exists at: {downloaded_model_path}")
    else:
        logging.error(f"Model file does NOT exist at: {downloaded_model_path}")

    # Load K-Means
    logging.info('Loading model')
    kmeans = joblib.load(downloaded_model_path)
    
    # Load scaler artifact
    logging.info('Downloading scaler artifact')
    scaler_artifact = run.use_artifact('tcc-ufrn/k-means-with-umap/scaler:latest', type='preprocessing')
    scaler_artifact_dir = scaler_artifact.download()

    # Verificar e imprimir o caminho do scaler
    scaler_path = os.path.join(scaler_artifact_dir, 'scaler.pkl')

    # Verificar se o arquivo existe
    if os.path.exists(scaler_path):
        logging.info(f"Scaler file exists at: {scaler_path}")
    else:
        logging.error(f"Scaler file does NOT exist at: {scaler_path}")

    # Load scaler
    logging.info('Loading scaler')
    scaler = joblib.load(scaler_path)

    # Load umap artifact
    logging.info('Downloading UMAP artifact')
    umap_artifact = run.use_artifact('tcc-ufrn/k-means-with-umap/umap:latest', type='model')
    umap_artifact_dir = umap_artifact.download()

    # Verificar e imprimir o caminho do umap
    umap_path = os.path.join(umap_artifact_dir, 'umap.pkl')

    # Verificar se o arquivo existe
    if os.path.exists(umap_path):
        logging.info(f"umap file exists at: {umap_path}")
    else:
        logging.error(f"umap file does NOT exist at: {umap_path}")

    # Load UMAP
    logging.info('Loading umap')
    umap = joblib.load(umap_path)

    # Create embedding matrix
    logging.info('Create embedding matrix')
    embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(embedding.values)

    # Normalize matrix
    logging.info('Normalize matrix')
    matrix_scaled = scaler.transform(matrix)

    # Dimensionality reduction
    two_dimensions = umap.transform(matrix_scaled)

    # Prever os clusters dos novos dados
    predicted_labels = kmeans.predict(two_dimensions)
    logging.info(f"Tamanho de predicted_labels: {len(predicted_labels)}")

    mapeamento = ["Vagas", "Eventos", "Informes", "Ciências"]

    # Dicionário de mapeamento
    label_mapping = {
        mapeamento[0]: 0,
        mapeamento[1]: 1,
        mapeamento[2]: 2,
        mapeamento[3]: 3
    }

    # Aplicar o mapeamento às labels
    true_labels = classified_news['target'].map(label_mapping)
    logging.info(f"Tamanho de true_labels: {len(true_labels)}")

    # Logar a matriz de confusão no W&B
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_labels,
        preds=predicted_labels,
        class_names=mapeamento
    )})

    # Acurácia
    accuracy = accuracy_score(true_labels, predicted_labels)
    logging.info(f"Accuracy: {accuracy}")
    wandb.log({"Accuracy": accuracy})

    # Finalizar a run
    wandb.finish()

main()
