import pandas as pd

df = pd.read_csv('data/first_news.csv')
header = df.iloc[0]
news_df = df.iloc[1:]

filtered_news_df = news_df[(news_df['target'] != 'Ciências') & 
                           (news_df['target'] != 'Vagas') & 
                           (news_df['target'] != 'Eventos') & 
                           (news_df['target'] != 'Informes')]

print(filtered_news_df)

count_targets = news_df['target'].value_counts()

print(count_targets)

print(len(news_df['target']))
