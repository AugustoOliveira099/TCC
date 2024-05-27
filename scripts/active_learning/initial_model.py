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

from utils import combine_columns, \
                  evaluate_model, \
                  classify_new_texts, \
                  predictions_to_dataframe, \
                  clean_text, \
                  remove_stopwords, \
                  lemmatize

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

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
X_train, X_test, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.1, random_state=42)

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

    # Verifica qual o melhor modelo com base na acurácia do modelo de teste
    if (test_accuracy > greater_accuracy):
        best_model = model
        greater_accuracy = test_accuracy

# Lê arquivos csv com os dados das notícias
logging.info('Read all news')
df1 = pd.read_csv('../../data/first_part.csv')
df2 = pd.read_csv('../../data/second_part.csv')

# Faz o merge dos dataframes
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

# Combina colunas do dataframe
df_full_combined = combine_columns(df_merged, 'title', 'content', 'combined')

# Classifica notícias
logging.info('Classifies news')
confident_predictions = classify_new_texts(df_full_combined['combined'], best_model, vectorizer, label_encoder, 0.7)

# Transforms data into dataframe
df_confident_predictions = predictions_to_dataframe(confident_predictions)

print(df_confident_predictions['target'].value_counts())

# Save data as csv file
df_confident_predictions.to_csv('../../data/all_classified_news.csv', index=False)
