"""
Script com funções que serão utilizadas 
"""

# Importa bibliotecas
import pandas as pd
import numpy as np
import logging
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Baixa recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# create OpenAi client
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def plot_frequency_of_tokens(df: pd.DataFrame) -> None:
    logging.info('Plot the token frequency')
    # Set the style of the plot
    plt.style.use('ggplot')

    # Create the histogram using the 'Token' column
    plt.hist(df['n_tokens'], bins=30, edgecolor='white')

    # Add labels for x and y axes
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')

    # Calculate the mean of the 'n_tokens' column
    mean_value = df['n_tokens'].mean()

    # Add a vertical line for the mean value
    plt.axvline(mean_value, color='blue',
                linestyle='dashed',
                linewidth=1,
                label=f'Mean: {mean_value:.2f}')

    # Find the maximum value of the x-axis
    max_x_value = df['n_tokens'].max()

    # Add an arrow pointing to the maximum x value
    plt.annotate(f'Max: {max_x_value}',
                 xy=(max_x_value, 0),
                 xycoords='data',
                 xytext=(max_x_value - 30, 25),
                 textcoords='data',
                 arrowprops=dict(arrowstyle='->', lw=1.5),
                 fontsize=10)

    # Add the legend
    plt.legend()

    # Save the plot as an image file (e.g., 'histogram.png')
    plt.savefig('histogram.png')

    # Close the plot
    plt.close()

# Função para remover HTML de uma string
def remove_html(text) -> str:
    soup = BeautifulSoup(text, "html.parser")
    # Remover todas as tags 'figcaption' e seu conteúdo
    for figcaption in soup.find_all('figcaption'):
        figcaption.decompose()
    # Remover linhas em branco e retornar o texto sem HTML
    return '\n'.join([line for line in soup.get_text().split('\n') if line.strip()])

# Remove HTML do texto
def remove_html_from_df(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    logging.info(f'Remove HTML from column {target_column}')
    df[target_column] = df[target_column].apply(lambda x: remove_html(x))
    return df

# Combina colunas de um dataframe
def combine_columns(df: pd.DataFrame,
                    title_column: str,
                    content_column: str,
                    result_column: str = 'combined') -> pd.DataFrame:
    # Combina os dados da coluna tile com os dados da coluna content
    logging.info('Combine data')
    df[result_column] = 'Título: ' + df[title_column].str.strip() + '; Conteúdo: ' + df[content_column].str.strip()
    return df

# Calculate the number of tokens
def calc_number_tokens(df: pd.DataFrame,
                       target_column: str,
                       result_column: str,
                       embedding_encoding: str) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(embedding_encoding)
    df[result_column] = df[target_column].apply(lambda x: len(encoding.encode(x)))
    return df

# Remove news that are too long to embed
def remove_long_news(df: pd.DataFrame, column_tokens: str, max_tokens: int) -> pd.DataFrame:
    # Remove news that are too long to embed
    sum_news_too_long = (df[column_tokens] > max_tokens).sum()
    logging.info(f'{sum_news_too_long} news are too long to embed')
    df = df[df[column_tokens] <= max_tokens]
    return df

# Requisita embedding à API da OpenAi
def get_embedding(text: str, model="text-embedding-3-large") -> list[float]:
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Evaluate model
def evaluate_model(model: XGBClassifier,
                   X_data: np.ndarray,
                   y_data: pd.Series,
                   label_encoder: LabelEncoder | None = None,
                   set_name: str = "") -> float:
    # Fazer previsões no conjunto
    y_pred = model.predict(X_data)

    # Avaliar a performance do modelo
    accuracy = accuracy_score(y_data, y_pred)
    if (label_encoder != None):
        report = classification_report(y_data, y_pred, target_names=label_encoder.classes_)
    else:
        report = classification_report(y_data, y_pred)

    logging.info(f'Accuracy {set_name}: {accuracy:.5f}')
    logging.info(f'Classification Report {set_name}:\n{report}')

    return accuracy

# Função para remover acentos
def remove_accents(text):
    # Normalizar o texto para decompor caracteres compostos
    text = unicodedata.normalize('NFKD', text)
    # Remove caracteres diacríticos (combining characters)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

# Função para limpar o texto
def clean_text(text: str) -> str:
    text = text.lower()  # Converter para minúsculas
    # text = text.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    text = remove_accents(text)  # Remover acentos
    text = remove_html(text) # Remove HTML
    text = nltk.word_tokenize(text, language='portuguese')  # Tokenizar
    return text

# Função para remover stopwords
stop_words = set(stopwords.words('portuguese'))
def remove_stopwords(tokens) -> list:
    return [word for word in tokens if word not in stop_words]

# Função para lematizar
nlp = spacy.load('pt_core_news_sm')  # Carregar o modelo de linguagem do spaCy para português
def lemmatize(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def classify_new_texts(new_texts: pd.Series,
                       model: XGBClassifier,
                       vectorizer: TfidfVectorizer,
                       label_encoder: LabelEncoder,
                       threshold: float = 0.7) -> list:
    # Pré-processar os novos textos
    processed_texts = new_texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)
    processed_texts = processed_texts.apply(lambda x: ' '.join(x))
    
    # Transformar os textos em vetores numéricos usando TF-IDF
    X_new = vectorizer.transform(processed_texts)
    
    # Normalizar os vetores TF-IDF
    normalizer = Normalizer()
    X = normalizer.fit_transform(X_new)
    
    # Obter as probabilidades das classes
    probas = model.predict_proba(X_new)
    
    # Filtra previsões com probabilidade maior que o limiar
    confident_preds = []
    for i, prob in enumerate(probas):
        max_prob = np.max(prob)
        if max_prob >= threshold:
            pred_label = label_encoder.inverse_transform([np.argmax(prob)])[0]
            confident_preds.append((new_texts.iloc[i], pred_label, max_prob))
    
    return confident_preds

# Transforma confident_predictions em um DataFrame
def predictions_to_dataframe(predictions):
    # Cria um DataFrame a partir das previsões confiáveis
    df = pd.DataFrame(predictions, columns=['combined', 'target', 'probability'])
    return df
