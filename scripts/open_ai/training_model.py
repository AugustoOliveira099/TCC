import pandas as pd
import logging
import numpy as np
import sys
import os
import wandb
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from ast import literal_eval
from dotenv import load_dotenv

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.training import evaluate_model

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')


def main() -> None:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="open-ai-model")

    logging.info('Read data')
    datafile_path = '../../data/noticias_ufrn_clusters.csv'
    df = pd.read_csv(datafile_path)

    # Lê a matriz de valores com redução de dimensionalidade
    matrix_tsne = df[['tsne1', 'tsne2']].values

    # Lê os rótulos gerados pelo KMeans com t-SNE
    cluster_column = 'cluster_with_tsne'
    labels = df[cluster_column]

    logging.info('Data split')
    # Dividir os dados em conjuntos de treinamento e teste 80/20
    X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(matrix_tsne, labels, test_size=0.2, random_state=42)

    # Dividir os dados de treinamento em treinamento e dev, totalizando 80/10/10
    X_test_tsne, X_dev_tsne, y_test_tsne, y_dev_tsne = train_test_split(X_test_tsne, y_test_tsne, test_size=0.5, random_state=42)

    # Treina o modelo XGBoost
    logging.info('Train the XGBoost model with tsne data')
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train_tsne, y_train_tsne)

    # Evaluate train model
    evaluate_model(model, X_train_tsne, y_train_tsne, "tsne train")

    # Evaluate dev model
    evaluate_model(model, X_dev_tsne, y_dev_tsne, "tsne dev")

    # Evaluate test model
    evaluate_model(model, X_test_tsne, y_test_tsne, "tsne test")

    # Create embedding matrix
    logging.info('Create embedding matrix')
    embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(embedding.values)

    # Normalizes matrix
    logging.info('Normalize matrix')
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # Save scaler
    scaler_path = 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    artifact = wandb.Artifact(name="scaler", type="preprocessing")
    artifact.add_file(scaler_path, name="scaler.pkl")
    wandb.log_artifact(artifact)

    # Selecionar os rótulos gerados pelo KMeans sem t-SNE
    cluster_column = 'cluster_without_tsne'
    labels = df[cluster_column]

    logging.info('Data split')

    # Dividir os dados em conjuntos de treinamento e teste 90/10
    X_train, X_test, y_train, y_test = train_test_split(matrix_scaled, labels, test_size=0.1, random_state=42)

    # Dividir os dados de treinamento em treinamento e dev, totalizando 90/5/5
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # param_grid = {
    #     'learning_rate': [0.3, 0.2, 0.3],
    #     'max_depth': [7, 6, 6],
    #     'reg_alpha': [15, 15, 10],
    #     'reg_lambda': [15, 15, 10],
    #     'subsample': [0.5, 0.5, 0.5],
    #     'colsample_bytree': [0.7, 0.7, 0.5],
    # }

    logging.info('Train the XGBoost model without tsne data')
    model = XGBClassifier(learning_rate=0.2,
                        max_depth=6,
                        reg_alpha=15,
                        reg_lambda=15,
                        subsample=0.5,
                        colsample_bytree=0.7,
                        eval_metric='mlogloss',
                        random_state=42,
                        objective='multi:softmax',
                        num_class=4)
    model.fit(X_train, y_train)

    # Evaluate train model
    evaluate_model(model, X_train, y_train, "train")

    # Evaluate dev model
    evaluate_model(model, X_dev, y_dev, "dev")

    # Evaluate test model
    evaluate_model(model, X_test, y_test, "test")

    # model.save_model('xgboost_model.json')

    # artifact = wandb.Artifact('xgboost_model', type='model')
    # artifact.add_file('xgboost_model.json')
    # wandb.log_artifact(artifact)
    
    wandb.finish()

main()
