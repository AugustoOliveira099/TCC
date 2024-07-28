import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Baixa recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

def remove_html(text: str) -> str:
    """
    Remove html from a given text.

    Args:
        text (str): The text that will have the HTML removed.

    Returns:
        str: Text without HTML.
    """
    soup = BeautifulSoup(text, "html.parser")
    # Removes all 'figcaption' tags and their content
    for figcaption in soup.find_all('figcaption'):
        figcaption.decompose()
    # Removes blank lines and returns the text without HTML
    return '\n'.join([line for line in soup.get_text().split('\n') if line.strip()])

def remove_accents(text):
    """
    Removes accents from the input text.

    Args:
        text (str): The text from which to remove accents.

    Returns:
        str: The text with accents removed.
    """
    # Normalize the text to decompose composed characters
    text = unicodedata.normalize('NFKD', text)
    # Remove diacritical marks (combining characters)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

def clean_text(text: str) -> str:
    """
    Cleans the input text by performing several preprocessing steps.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    # text = text.translate(str.maketrans('', '', string.punctuation))  # Remover pontuação
    text = remove_accents(text)  # Remove accents
    text = remove_html(text) # Remove HTML
    text = nltk.word_tokenize(text, language='portuguese')  # Tokenize
    return text

stop_words = set(stopwords.words('portuguese'))
def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes stopwords from a list of tokens.

    Args:
        tokens (List[str]): The list of tokens from which to remove stopwords.

    Returns:
        List[str]: A list of tokens with stopwords removed.
    """
    return [word for word in tokens if word not in stop_words]

nlp = spacy.load('pt_core_news_sm')  # Load the spaCy language model for portuguese
def lemmatize(tokens):
    """
    Lemmatizes a list of tokens using a spaCy model.

    Args:
        tokens (List[str]): The list of tokens to lemmatize.

    Returns:
        List[str]: A list of lemmatized tokens.
    """
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def remove_long_news(df: pd.DataFrame,
                     column_tokens: str,
                     max_tokens: int) -> pd.DataFrame:
    """
    Removes rows from the DataFrame where the number of tokens in the specified 
        column exceeds the maximum allowed.

    Args:
        df (pd.DataFrame): The DataFrame containing the news data.
        column_tokens (str): The name of the column containing the token counts.
        max_tokens (int): The maximum number of tokens allowed for each news item.

    Returns:
        pd.DataFrame: The filtered DataFrame with news items that have token counts 
            within the specified limit.
    """
    # Remove news that are too long to embed
    sum_news_too_long = (df[column_tokens] > max_tokens).sum()
    logging.info(f'{sum_news_too_long} news are too long to embed')
    df = df[df[column_tokens] <= max_tokens]
    return df

def combine_columns(df: pd.DataFrame,
                    title_column: str,
                    content_column: str,
                    result_column: str = 'combined') -> pd.DataFrame:
    """
    Combines the title and news content into one column.

    Args:
        df (pd.DataFrame): Dataframe with data.
        title_column (str): Column name with news title.
        content_column (str): Column name with news content.
        result_column (str): Resulting column name.

    Returns:
        pd.DataFrame: Dataframe with combined columns.
    """
    df.loc[:, result_column] = 'Título: ' + df[title_column].str.strip() + '; Conteúdo: ' + df[content_column].str.strip()
    return df

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return ' '.join(text)
