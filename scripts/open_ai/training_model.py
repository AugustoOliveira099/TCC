import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
import numpy as np
from ast import literal_eval

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Load data
datafile_path = '../../data/noticias_ufrn_clusters.csv'
df = pd.read_csv(datafile_path)

# Create embedding matrix
logging.info('Create embedding matrix (this may take about 10 minutes)')
df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embedding.values)

# Normalizar a matriz de embeddings
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix)

# Seleciona coluna com as labels
cluster_column = 'cluster_with_tsne'
labels = df[cluster_column]

# Dividir os dados em conjuntos de treinamento e teste 80/20
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(matrix, labels, test_size=0.2, random_state=42)

# Dividir os dados de treinamento em treinamento e dev, totalizando 80/10/10
X_train_tsne, X_dev_tsne, y_train_tsne, y_dev_tsne = train_test_split(X_test_tsne, y_test_tsne, test_size=0.5, random_state=42)

# Seleciona coluna com as labels
cluster_column = 'cluster_without_tsne'
labels = df[cluster_column]

# Dividir os dados em conjuntos de treinamento e teste 80/20
X_train, X_test, y_train, y_test = train_test_split(matrix, labels, test_size=0.2, random_state=42)

# Dividir os dados de treinamento em treinamento e dev, totalizando 80/10/10
X_train, X_dev, y_train, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)