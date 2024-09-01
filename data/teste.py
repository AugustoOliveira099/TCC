import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
# # Kmeans
# report = {
#     "Vagas": {"Precisão": 0.94, "Recall": 0.91, "F1-score": 0.92},
#     "Ciências": {"Precisão": 0.51, "Recall": 0.82, "F1-score": 0.63},
#     "Eventos": {"Precisão": 0.71, "Recall": 0.80, "F1-score": 0.75},
#     "Informes": {"Precisão": 0.81, "Recall": 0.43, "F1-score": 0.56},
#     "Acurácia": {"Precisão": 0.72, "Recall": 0.72, "F1-score": 0.72}
# }


# # Kmeans PCA
# report = {
#     'Informes': {
#         'Precisão': 0.56,
#         'Recall': 0.55,
#         'F1-score': 0.55
#     },
#     'Vagas': {
#         'Precisão': 0.90,
#         'Recall': 0.91,
#         'F1-score': 0.90
#     },
#     'Eventos': {
#         'Precisão': 0.78,
#         'Recall': 0.67,
#         'F1-score': 0.72
#     },
#     'Ciências': {
#         'Precisão': 0.29,
#         'Recall': 0.34,
#         'F1-score': 0.31
#     },
#     'Acurácia': {
#         'Precisão': 0.62,
#         'Recall': 0.62,
#         'F1-score': 0.62
#     }
# }


# # XGBoost inicial
# report = {
#     'Ciências': {
#         'Precisão': 0.75,
#         'Recall': 0.71,
#         'F1-score': 0.73
#     },
#     'Eventos': {
#         'Precisão': 0.87,
#         'Recall': 0.74,
#         'F1-score': 0.80
#     },
#     'Informes': {
#         'Precisão': 0.68,
#         'Recall': 0.81,
#         'F1-score': 0.74
#     },
#     'Vagas': {
#         'Precisão': 0.92,
#         'Recall': 0.85,
#         'F1-score': 0.88
#     },
#     # 'Acurácia': {
#     #     'Precisão': 0.78,
#     #     'Recall': 0.78,
#     #     'F1-score': 0.78
#     # },
# }


# # XGBoost Final
# report = {
#     'Ciências': {
#         'Precisão': 0.81,
#         'Recall': 0.66,
#         'F1-score': 0.73
#     },
#     'Eventos': {
#         'Precisão': 0.75,
#         'Recall': 0.88,
#         'F1-score': 0.81
#     },
#     'Informes': {
#         'Precisão': 0.76,
#         'Recall': 0.78,
#         'F1-score': 0.77
#     },
#     'Vagas': {
#         'Precisão': 0.95,
#         'Recall': 0.92,
#         'F1-score': 0.93
#     }
#     'Acurácia': {
#         'Precisão': 0.81,
#         'Recall': 0.81,
#         'F1-score': 0.81
#     }
# }

report = {
    "Ciências": {"Precisão": 0.96, "Recall": 0.94, "F1-score": 0.95},
    "Eventos": {"Precisão": 0.96, "Recall": 0.98, "F1-score": 0.97},
    "Informes": {"Precisão": 0.98, "Recall": 0.98, "F1-score": 0.98},
    "Vagas": {"Precisão": 0.99, "Recall": 0.99, "F1-score": 0.99},
}



# Calculate the classification report
df_report = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 6))
sns.heatmap(df_report, annot=True, cmap="Blues")
image_path = '../images/report_xgboost_teste.png'
plt.savefig(image_path)


# df = pd.read_csv("noticias_ufrn_embeddings.csv")
# df1 = pd.read_csv("classified_news.csv")
# df2 = pd.read_csv("xgboost/noticias_xgboost.csv")
# df3 = pd.read_csv("news.csv")

# print(df.shape)
# print(df1.shape)
# print(df2.shape)
# print(df3.shape)

# # Plotar o histograma
# n, bins, patches = plt.hist(df["target"], bins=4, edgecolor='black', align='mid', alpha=0.7, rwidth=0.8)

# plt.xlabel('Classificação')
# plt.ylabel('Frequência')

# plt.ylim(0, max(n) + 800)

# labels = ['Ciências', 'Informes', 'Vagas', 'Eventos']
# plt.xticks([patch.get_x() + patch.get_width() / 2 for patch in patches], labels)

# for count, patch in zip(n, patches):
#     height = patch.get_height()
#     plt.text(patch.get_x() + patch.get_width() / 2, height + 70, int(count), ha='center')

# plt.show()



# # Save new dataset into Weights and Biases
# dataset_path = 'kmeans/noticias_kmeans.csv'
# run = wandb.init()
# logged_artifact = run.log_artifact(
#     dataset_path,
#     name="dataset_kmeans",
#     type="dataset"
# )
# run.link_artifact(
#     artifact=logged_artifact,
#     target_path="mlops2023-2-org/wandb-registry-dataset/dataset_kmeans"
# ) # Log and link the dataset to the Model Registry

# run.finish()
