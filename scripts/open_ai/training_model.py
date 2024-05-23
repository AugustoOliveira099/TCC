import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging
import numpy as np
from ast import literal_eval

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Load data
datafile_path = 'data/noticias_ufrn_clusters.csv'
df = pd.read_csv(datafile_path)

# Create embedding matrix
logging.info('Create embedding matrix (this may take about 10 minutes)')
df['embedding'] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embedding.values)
