import sys
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys
import os

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.training import evaluate_model

from utils.preprocessing import lemmatize, \
                                 remove_stopwords, \
                                 clean_text

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main() -> None:
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

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.2, random_state=42)

    # Train model
    logging.info('Train the XGBoost model')
    model = XGBClassifier(learning_rate=0.3,
                        max_depth=7,
                        reg_alpha=15,
                        reg_lambda=15,
                        subsample=0.5,
                        colsample_bytree=0.7,
                        eval_metric='mlogloss',
                        random_state=42)
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate train model
    evaluate_model(model, X_train, y_train, label_encoder, "train model")

    # Evaluate test model
    evaluate_model(model, X_test, y_test, label_encoder, "test model")
