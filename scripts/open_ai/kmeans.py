import pandas as pd
import logging
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('Read data')
datafile_path = '../../data/noticias_ufrn_embeddings.csv'
df = pd.read_csv(datafile_path)

# Create embedding matrix
logging.info('Create embedding matrix (this may take about 5 minutes)')
initial_time = time.time()
embedding = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(embedding.values)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to create embedding matrix was {total_time} seconds.')

# Normalize matrix
logging.info('Normalize matrix')
scaler = StandardScaler()
matrix_scaled = scaler.fit_transform(matrix)

# KMeans without t-SNE
logging.info('Init KMeans without t-SNE')
initial_time = time.time()
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
kmeans.fit(matrix_scaled)
labels = kmeans.labels_
df['cluster_without_tsne'] = labels
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to KMeans without t-SNE was {total_time} seconds.')

# Apply t-SNE to reduce dimensionality
logging.info('Apply t-SNE')
initial_time = time.time()
tsne_new = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=200)
matrix_tsne = tsne_new.fit_transform(matrix_scaled)
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time to apply t-SNE was {total_time} seconds.')

# KMeans with t-SNE
logging.info('Init KMeans with t-SNE')
initial_time = time.time()
kmeans_tsne = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
kmeans_tsne.fit(matrix_tsne)
labels_tsne = kmeans_tsne.labels_
df['cluster_with_tsne'] = labels_tsne
final_time = time.time()
total_time = final_time - initial_time
logging.info(f'The total time for KMeans with t-SNE was {total_time} seconds.')

# Save the results to a CSV file
logging.info('Save data with clusters')
df.drop(['title', 'content', 'combined', 'n_tokens', 'target'], axis=1, inplace=True)
tsne_df = pd.DataFrame(matrix_tsne, columns=['tsne1', 'tsne2'])
df = df.reset_index(drop=True)  # Reiniciar os índices do DataFrame original
df = pd.concat([df, tsne_df], axis=1)
df.to_csv('../../data/noticias_ufrn_clusters.csv', index=False)

# # Visualize the KMeans clusters with t-SNE
# plt.figure(figsize=(12, 6))
# x = [x for x, y in matrix_tsne]
# y = [y for x, y in matrix_tsne]
# for category, color in enumerate(['purple', 'green', 'red', 'blue']):
#     xs = np.array(x)[df.cluster_with_tsne == category]
#     ys = np.array(y)[df.cluster_with_tsne == category]
#     plt.scatter(xs, ys, color=color, alpha=0.3)
#     avg_x = xs.mean()
#     avg_y = ys.mean()

#     if color == 'purple':
#         marker_color = 'yellow'
#     elif color == 'green':
#         marker_color = 'orange'
#     elif color == 'red':
#         marker_color = 'cyan'
#     else:
#         marker_color = 'magenta'

#     plt.scatter(avg_x, avg_y, marker='x', color=marker_color, s=100)
# plt.title('KMeans Clusters with t-SNE')
# plt.savefig('../../images/clusters_with_tse.png')



# # Selecionar os rótulos gerados pelo KMeans com t-SNE
# cluster_column = 'cluster_with_tsne'
# labels = df[cluster_column]

# # Dividir os dados em conjuntos de treinamento e teste 80/20
# X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(matrix_tsne, labels, test_size=0.2, random_state=42)

# # Dividir os dados de treinamento em treinamento e dev, totalizando 80/10/10
# X_test_tsne, X_dev_tsne, y_test_tsne, y_dev_tsne = train_test_split(X_test_tsne, y_test_tsne, test_size=0.5, random_state=42)

# # Treinar o modelo XGBoost
# logging.info('Train the XGBoost model with tsne data')
# initial_time = time.time()
# model = XGBClassifier(eval_metric='mlogloss', random_state=42)
# model.fit(X_train_tsne, y_train_tsne)
# final_time = time.time()
# total_time = final_time - initial_time
# logging.info(f'The total time for train the model was {total_time} seconds.')

# # Fazer previsões no conjunto de treinamento
# y_pred_train = model.predict(X_train_tsne)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_train_tsne, y_pred_train)
# report = classification_report(y_train_tsne, y_pred_train)

# logging.info(f'Accuracy tsne train: {accuracy}')
# logging.info(f'Classification Report tsne train:\n{report}')

# # Fazer previsões no conjunto de dev
# y_pred_dev = model.predict(X_dev_tsne)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_dev_tsne, y_pred_dev)
# report = classification_report(y_dev_tsne, y_pred_dev)

# logging.info(f'Accuracy tsne dev: {accuracy}')
# logging.info(f'Classification Report tsne dev:\n{report}')

# # Fazer previsões no conjunto de teste
# y_pred_test = model.predict(X_test_tsne)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_test_tsne, y_pred_test)
# report = classification_report(y_test_tsne, y_pred_test)

# logging.info(f'Accuracy tsne test: {accuracy}')
# logging.info(f'Classification Report tsne test:\n{report}')




# # Selecionar os rótulos gerados pelo KMeans sem t-SNE
# cluster_column = 'cluster_without_tsne'
# labels = df[cluster_column]

# # Dividir os dados em conjuntos de treinamento e teste 90/10
# X_train, X_test, y_train, y_test = train_test_split(matrix_scaled, labels, test_size=0.1, random_state=42)

# # Dividir os dados de treinamento em treinamento e dev, totalizando 90/5/5
# X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# # Treinar o modelo XGBoost
# etas = [0.2, 0.1, 0.01]
# model = XGBClassifier(eval_metric='mlogloss', random_state=42)

# param_grid = {
#     'learning_rate': [0.2, 0.1],
#     'max_depth': [3, 5],
#     'reg_alpha': [1.0, 0.5],
#     'reg_lambda': [2, 0.5],
#     # 'subsample': [0.5, 0.7, 1.0],
#     # 'colsample_bytree': [0.8, 0.9, 1.0]
# }

# # Configurar o GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)

# logging.info('Train the XGBoost model without tsne data')
# initial_time = time.time()
# grid_search.fit(X_train, y_train)
# final_time = time.time()
# total_time = final_time - initial_time
# logging.info(f'The total time for train the model was {total_time} seconds.')

# # Obter os melhores parâmetros
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Fazer previsões no conjunto de treinamento
# y_pred_train = best_model.predict(X_train)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_train, y_pred_train)
# report = classification_report(y_train, y_pred_train)

# logging.info(f'Accuracy train: {accuracy}')
# logging.info(f'Classification Report train:\n{report}')

# # Fazer previsões no conjunto de dev
# y_pred_dev = best_model.predict(X_dev)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_dev, y_pred_dev)
# report = classification_report(y_dev, y_pred_dev)

# logging.info(f'Accuracy dev: {accuracy}')
# logging.info(f'Classification Report dev:\n{report}')

# # Fazer previsões no conjunto de teste
# y_pred_test = best_model.predict(X_test)

# # Avaliar a performance do modelo
# accuracy = accuracy_score(y_test, y_pred_test)
# report = classification_report(y_test, y_pred_test)

# logging.info(f'Accuracy test: {accuracy}')
# logging.info(f'Classification Report test:\n{report}')

# logging.info(f'Best Parameters: {best_params}')
