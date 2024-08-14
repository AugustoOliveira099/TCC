import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("noticias_ufrn_embeddings.csv")
df1 = pd.read_csv("classified_news.csv")
df2 = pd.read_csv("all_classified_news.csv")
df3 = pd.read_csv("news.csv")

print(df.shape)
print(df1.shape)
print(df2.shape)
print(df3.shape)

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
