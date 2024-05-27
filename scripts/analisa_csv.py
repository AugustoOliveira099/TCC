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

print(len(filtered_news_df['target']))
