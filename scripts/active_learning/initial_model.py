import sys
import os
import unicodedata
import string
import spacy
import logging
import numpy as np
import pandas as pd
import nltk
import time
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from ast import literal_eval

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.append(relative_path)

from utils import remove_html, \
                  combine_columns, \
                  evaluate_model

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Função para remover acentos
def remove_accents(text):
    # Normalizar o texto para decompor caracteres compostos
    text = unicodedata.normalize('NFKD', text)
    # Remove caracteres diacríticos (combining characters)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Remove HTML
    text = remove_html(text)
    return text

# Função para limpar o texto
def clean_text(text):
    text = text.lower()  # Converter para minúsculas
    # text = text.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    text = remove_accents(text)  # Remover acentos
    text = nltk.word_tokenize(text, language='portuguese')  # Tokenizar
    return text

# Função para remover stopwords
stop_words = set(stopwords.words('portuguese'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Carregar o modelo de linguagem do spaCy para português
nlp = spacy.load('pt_core_news_sm')

# Função para lematizar
def lemmatize(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

logging.info('Read data')
datafile_path = '../../data/classified_news.csv'
df = pd.read_csv(datafile_path)

# Combina colunas do dataframe
df_combined = combine_columns(df, 'title', 'content', 'combined')




# # Agrupar os dados por destino
# grouped = df_combined.groupby('target')

# # Lista para armazenar os DataFrames de destino balanceados
# balanced_dfs = []

# # Quantidade de instâncias desejadas
# desired_count = 250

# # Para cada grupo, selecionar aleatoriamente as instâncias desejadas
# for target, group in grouped:
#     if len(group) >= desired_count:
#         balanced_dfs.append(group.sample(desired_count))
#     else:
#         balanced_dfs.append(group)

# # Concatenar os DataFrames de destino balanceados em um único DataFrame
# balanced_data = pd.concat(balanced_dfs)

# # Remover a coluna de índice resultante da concatenação
# balanced_data.reset_index(drop=True, inplace=True)




# Separa os textos (features) e os targets
texts = df_combined['combined']
targets = df_combined['target']

# Converte categorias em códigos numéricos
label_encoder = LabelEncoder()
targets_encoded = label_encoder.fit_transform(targets)

# Crie um array de inteiros
array = np.array(targets_encoded)

# Transforme o array em um DataFrame
df = pd.DataFrame(array, columns=['Inteiros'])

# Mostra a correspondência entre códigos numéricos e classes
for code, category in enumerate(label_encoder.classes_):
    logging.info(f'Código {code}: {category}')

# Aplicar o pré-processamento aos textos
logging.info('Preprocessing data')
processed_texts = texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)

# Converter listas de tokens de volta para strings
logging.info('Preprocessing data')
processed_texts = processed_texts.apply(lambda x: ' '.join(x))

# Transformar os textos em vetores numéricos usando TF-IDF
logging.info('Vectorize data')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

logging.info('Train model')

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.1, random_state=42)

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
    evaluate_model(model, X_train, y_train, "train model")

    # Evaluate test model
    evaluate_model(model, X_test, y_test, "test model")
