import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys
import os

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.training import evaluate_model, \
                           classify_new_texts

from utils.preprocessing import lemmatize, \
                                remove_stopwords, \
                                clean_text, \
                                combine_columns

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main() -> None:
    logging.info('Read data')
    datafile_path = '../../data/classified_news.csv'
    df = pd.read_csv(datafile_path)

    # Combina colunas do dataframe
    logging.info('Combine columns')
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

    # Criar e treinar o modelo XGBoost
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
    evaluate_model(model, X_train, y_train, "train model", label_encoder)

    # Evaluate test model
    evaluate_model(model, X_test, y_test, "test model", label_encoder)

    # Lê arquivos csv com os dados das notícias
    logging.info('Reading news')
    all_news_df = pd.read_csv('../../data/news.csv')

    # Retira as notícias que foram classificadas manualmente
    filter_news_df = all_news_df[~all_news_df['content'].isin(df_combined['content'])]

    # Drop duplicates
    filter_news_df = filter_news_df.drop_duplicates(subset=['content'])

    # Combina colunas do dataframe
    logging.info('Combine columns')
    df_full_combined = combine_columns(filter_news_df, 'title', 'content', 'combined')

    # Classifica notícias
    logging.info('Classifies news')
    confident_predictions = classify_new_texts(df_full_combined['combined'], model, vectorizer, label_encoder, 0.8)

    # Transforms data into dataframe
    df_confident_predictions = pd.DataFrame(confident_predictions, columns=['combined', 'target', 'probability'])

    print(df_confident_predictions['target'].value_counts())

    # Save data as csv file
    df_confident_predictions.to_csv('../../data/all_classified_news.csv', index=False)

main()
