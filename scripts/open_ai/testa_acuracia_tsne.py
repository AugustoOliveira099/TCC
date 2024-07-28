import os
import pandas as pd
import logging
import numpy as np
import joblib
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
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
        project="k-means-with-tsne"
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
    downloaded_model_path = run.use_model('tcc-ufrn/model-registry/kmeans_tsne:latest')

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
    scaler_artifact = run.use_artifact('tcc-ufrn/k-means-with-tsne/scaler:latest', type='preprocessing')
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

    # Load tsne artifact
    logging.info('Downloading t-SNE artifact')
    tsne_artifact = run.use_artifact('tcc-ufrn/k-means-with-tsne/tsne:latest', type='model')
    tsne_artifact_dir = tsne_artifact.download()

    # Verificar e imprimir o caminho do tsne
    tsne_path = os.path.join(tsne_artifact_dir, 'tsne.pkl')

    # Verificar se o arquivo existe
    if os.path.exists(tsne_path):
        logging.info(f"tsne file exists at: {tsne_path}")
    else:
        logging.error(f"tsne file does NOT exist at: {tsne_path}")

    # Load tsne
    logging.info('Loading tsne')
    tsne = joblib.load(tsne_path)

    # Create embedding matrix
    logging.info('Create embedding matrix')
    embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(embedding.values)

    # Normalize matrix
    logging.info('Normalize matrix')
    matrix_scaled = scaler.transform(matrix)

    # Dimensionality reduction
    two_dimensions = tsne.transform(matrix_scaled)

    # Prever os clusters dos novos dados
    predicted_labels = kmeans.predict(two_dimensions)
    logging.info(f"Tamanho de predicted_labels: {len(predicted_labels)}")

    mapeamento = ["Eventos", "Informes", "Vagas", "Ciências"]

    # Dicionário de mapeamento
    label_mapping = {
        mapeamento[0]: 0,
        mapeamento[1]: 1,
        mapeamento[2]: 2,
        mapeamento[3]: 3
    }

    # Aplicar o mapeamento às labels
    true_labels = classified_news['target'].map(label_mapping)

    # Calcula a matriz de confusão
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Cria o heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=mapeamento, yticklabels=mapeamento)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    image_path = '../../images/heatmap_tsne.png'
    plt.savefig(image_path)

    # Logue o gráfico no W&B
    wandb.log({"confusion_matrix_heatmap": wandb.Image(image_path)})

    # Acurácia
    accuracy = accuracy_score(true_labels, predicted_labels)
    logging.info(f"Accuracy: {accuracy}")
    wandb.log({"Accuracy": accuracy})

    # Finalizar a run
    wandb.finish()

main()
