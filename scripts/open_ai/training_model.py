import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import logging
import numpy as np
import time
from ast import literal_eval

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('Read data')
datafile_path = '../../data/noticias_ufrn_clusters.csv'
df = pd.read_csv(datafile_path)

# Lê a matriz de valores com redução de dimensionalidade
matrix_tsne = df[['tsne1', 'tsne2']].values

print(type(matrix_tsne))

# Lê os rótulos gerados pelo KMeans com t-SNE
cluster_column = 'cluster_with_tsne'
labels = df[cluster_column]

logging.info('Data split')
# Dividir os dados em conjuntos de treinamento e teste 80/20
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(matrix_tsne, labels, test_size=0.2, random_state=42)

# Dividir os dados de treinamento em treinamento e dev, totalizando 80/10/10
X_test_tsne, X_dev_tsne, y_test_tsne, y_dev_tsne = train_test_split(X_test_tsne, y_test_tsne, test_size=0.5, random_state=42)

# Treinar o modelo XGBoost
logging.info('Train the XGBoost model with tsne data')
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train_tsne, y_train_tsne)

# Fazer previsões no conjunto de treinamento
y_pred_train = model.predict(X_train_tsne)

# Avaliar a performance do modelo
accuracy = accuracy_score(y_train_tsne, y_pred_train)
report = classification_report(y_train_tsne, y_pred_train)

logging.info(f'Accuracy tsne train: {accuracy}')
logging.info(f'Classification Report tsne train:\n{report}')

# Fazer previsões no conjunto de dev
y_pred_dev = model.predict(X_dev_tsne)

# Avaliar a performance do modelo
accuracy = accuracy_score(y_dev_tsne, y_pred_dev)
report = classification_report(y_dev_tsne, y_pred_dev)

logging.info(f'Accuracy tsne dev: {accuracy}')
logging.info(f'Classification Report tsne dev:\n{report}')

# Fazer previsões no conjunto de teste
y_pred_test = model.predict(X_test_tsne)

# Avaliar a performance do modelo
accuracy = accuracy_score(y_test_tsne, y_pred_test)
report = classification_report(y_test_tsne, y_pred_test)

logging.info(f'Accuracy tsne test: {accuracy}')
logging.info(f'Classification Report tsne test:\n{report}')



# Create embedding matrix
logging.info('Create embedding matrix')
initial_time = time.time()
embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(embedding.values)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time for create embedding matrix was {total_time} seconds.')

# Normalize matrix
logging.info('Normalize matrix')
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix)

# Selecionar os rótulos gerados pelo KMeans sem t-SNE
cluster_column = 'cluster_without_tsne'
labels = df[cluster_column]

logging.info('Data split')

# Dividir os dados em conjuntos de treinamento e teste 90/10
X_train, X_test, y_train, y_test = train_test_split(matrix_scaled, labels, test_size=0.1, random_state=42)

# Dividir os dados de treinamento em treinamento e dev, totalizando 90/5/5
X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# # Estancia modelos e define hiperparâmetros
# logging.info('Create model')

param_grid = {
    'learning_rate': [0.3, 0.2, 0.3],
    'max_depth': [7, 6, 6],
    'reg_alpha': [15, 15, 10],
    'reg_lambda': [15, 15, 10],
    'subsample': [0.5, 0.5, 0.5],
    'colsample_bytree': [0.7, 0.7, 0.5],
}

# # Configurar o GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)

for i in range(3):
    # model fit
    logging.info('Train the XGBoost model without tsne data')
    logging.info(f'Parameters: { {'learning_rate': param_grid['learning_rate'][i], 
                                  'max_depth': param_grid['max_depth'][i], 
                                  'reg_alpha': param_grid['reg_alpha'][i],
                                  'reg_lambda': param_grid['reg_lambda'][i], 
                                  'subsample': param_grid['subsample'][i],
                                  'colsample_bytree': param_grid['colsample_bytree'][i],
                                  } }')
    initial_time = time.time()
    model = XGBClassifier(learning_rate=param_grid['learning_rate'][i],
                        max_depth=param_grid['max_depth'][i],
                        reg_alpha=param_grid['reg_alpha'][i],
                        reg_lambda=param_grid['reg_lambda'][i],
                        subsample=param_grid['subsample'][i],
                        colsample_bytree=param_grid['colsample_bytree'][i],
                        eval_metric='mlogloss',
                        random_state=42)
    model.fit(X_train, y_train)
    final_time = time.time()
    total_time = final_time - initial_time
    logging.info(f'The total time for train the model was {total_time} seconds.')

    # # Obter os melhores parâmetros
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # Fazer previsões no conjunto de treinamento
    y_pred_train = model.predict(X_train)

    # Avaliar a performance do modelo
    accuracy = accuracy_score(y_train, y_pred_train)
    report = classification_report(y_train, y_pred_train)

    logging.info(f'Accuracy train: {accuracy}')
    logging.info(f'Classification Report train:\n{report}')

    # Fazer previsões no conjunto de dev
    y_pred_dev = model.predict(X_dev)

    # Avaliar a performance do modelo
    accuracy = accuracy_score(y_dev, y_pred_dev)
    report = classification_report(y_dev, y_pred_dev)

    logging.info(f'Accuracy dev: {accuracy}')
    logging.info(f'Classification Report dev:\n{report}')

    # Fazer previsões no conjunto de teste
    y_pred_test = model.predict(X_test)

    # Avaliar a performance do modelo
    accuracy = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test)

    logging.info(f'Accuracy test: {accuracy}')
    logging.info(f'Classification Report test:\n{report}')
