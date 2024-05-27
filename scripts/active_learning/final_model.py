import sys
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.append(relative_path)

from utils import evaluate_model, \
                  clean_text, \
                  remove_stopwords, \
                  lemmatize

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('Read data')
datafile_path = '../../data/all_classified_news.csv'
df = pd.read_csv(datafile_path)

# Separa os textos (features) e os targets
texts = df['combined']
targets = df['target']

# Converte categorias em códigos numéricos
label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets)

# Mostra a correspondência entre códigos numéricos e classes
for code, category in enumerate(label_encoder.classes_):
    logging.info(f'Código {code}: {category}')

# Aplica o pré-processamento aos textos
logging.info('Data preprocessing')
processed_texts = texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)

# Converte listas de tokens de volta para strings
logging.info('Convert list of tokens to string')
processed_texts = processed_texts.apply(lambda x: ' '.join(x))

# Transforma os textos em vetores numéricos usando TF-IDF
logging.info('Vectorize data')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

logging.info('Train model')

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.2, random_state=42)

# Instancia variável que contém o melhor modelo testado
best_model = XGBClassifier()
greater_accuracy = 0

param_grid = {
    'learning_rate': [0.3, 0.2, 0.01],
    'max_depth': [7, 6, 1],
    'reg_alpha': [15, 15, 300],
    'reg_lambda': [15, 15, 400],
    'subsample': [0.5, 0.5, 0.1],
    'colsample_bytree': [0.7, 0.7, 0.1],
}

for i in range(3):
    logging.info('Train the XGBoost model')
    logging.info(f'Parameters: { {'learning_rate': param_grid['learning_rate'][i],
                                  'max_depth': param_grid['max_depth'][i],
                                  'reg_alpha': param_grid['reg_alpha'][i],
                                  'reg_lambda': param_grid['reg_lambda'][i],
                                  'subsample': param_grid['subsample'][i],
                                  'colsample_bytree': param_grid['colsample_bytree'][i],
                                  } }')
    # Criar e treinar o modelo XGBoost
    model = XGBClassifier(learning_rate=param_grid['learning_rate'][i],
                        max_depth=param_grid['max_depth'][i],
                        reg_alpha=param_grid['reg_alpha'][i],
                        reg_lambda=param_grid['reg_lambda'][i],
                        subsample=param_grid['subsample'][i],
                        colsample_bytree=param_grid['colsample_bytree'][i],
                        eval_metric='mlogloss',
                        random_state=42)
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate train model
    evaluate_model(model, X_train, y_train, label_encoder, "train model")

    # Evaluate test model
    test_accuracy = evaluate_model(model, X_test, y_test, label_encoder, "test model")
