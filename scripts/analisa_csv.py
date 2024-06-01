import pandas as pd

df = pd.read_csv('../data/all_classified_news.csv')
header = df.iloc[0]
news_df = df.iloc[1:]

filtered_news_df = news_df[(news_df['target'] != 'Ciências') & 
                           (news_df['target'] != 'Vagas') & 
                           (news_df['target'] != 'Eventos') & 
                           (news_df['target'] != 'Informes')]

filtered_news_df = news_df[(news_df['probability'] >= 0.8)]

# count_targets = news_df['target'].value_counts()

# print(count_targets)

# print(len(filtered_news_df['target']))

df1 = pd.read_csv('../data/all_classified_news.csv')
df2 = pd.read_csv('../data/noticias_ufrn_clusters.csv')

coluna_df1 = df1[['combined']]
colunas_df2 = df2[['cluster_without_tsne', 'cluster_with_tsne']]

novo_df = pd.concat([coluna_df1, colunas_df2], axis=1)

novo_df.to_csv('analisa_resultados.csv', index=False)
