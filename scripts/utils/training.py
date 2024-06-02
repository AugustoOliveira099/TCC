import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocessing import clean_text, \
                                remove_stopwords, \
                                lemmatize

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def classify_new_texts(new_texts: pd.Series,
                       model: XGBClassifier,
                       vectorizer: TfidfVectorizer,
                       label_encoder: LabelEncoder,
                       threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """
    Classifies new texts using a pre-trained XGBClassifier model, TF-IDF vectorizer, and label encoder.
    
    Args:
        new_texts (pd.Series): Series containing the new texts to be classified.
        model (XGBClassifier): Pre-trained XGBClassifier model for classification.
        vectorizer (TfidfVectorizer): Pre-fitted TF-IDF vectorizer for transforming texts.
        label_encoder (LabelEncoder): Pre-fitted label encoder for decoding labels.
        threshold (float, optional): Probability threshold for confident predictions. Defaults to 0.7.
    
    Returns:
        List[Tuple[str, str, float]]: A list of tuples containing the text, predicted label, and prediction probability for texts with a confidence greater than the threshold.
    """
    # Pre-process the new texts
    processed_texts = new_texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)
    processed_texts = processed_texts.apply(lambda x: ' '.join(x))
    
    # Transform the texts into numerical vectors using TF-IDF
    X_new = vectorizer.transform(processed_texts)
    
    # Normalize the TF-IDF vectors
    normalizer = Normalizer()
    X = normalizer.fit_transform(X_new)
    
    # Obtain class probabilities
    probas = model.predict_proba(X_new)
    
    # Filter predictions with probability higher than the threshold
    confident_preds = []
    for i, prob in enumerate(probas):
        max_prob = np.max(prob)
        if max_prob >= threshold:
            pred_label = label_encoder.inverse_transform([np.argmax(prob)])[0]
            confident_preds.append((new_texts.iloc[i], pred_label, max_prob))
    
    return confident_preds

def evaluate_model(model: XGBClassifier,
                   X_data: np.ndarray,
                   y_data: pd.Series,
                   set_name: str = "",
                   label_encoder: LabelEncoder | None = None) -> float:
    """
    Evaluates the performance of a given XGBClassifier model on the provided dataset.

    Args:
        model (XGBClassifier): The model to evaluate.
        X_data (np.ndarray): The feature data to use for predictions.
        y_data (pd.Series): The true labels corresponding to the feature data.
        set_name (str, optional): The name of the dataset being evaluated (e.g., 'train', 'test'). Defaults to an empty string.
        label_encoder (LabelEncoder, optional): A label encoder for decoding class labels. Defaults to None.

    Returns:
        float: The accuracy of the model on the provided dataset.
    """
    # Make predictions on the dataset
    y_pred = model.predict(X_data)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_data, y_pred)
    if (label_encoder != None):
        report = classification_report(y_data, y_pred, target_names=label_encoder.classes_)
    else:
        report = classification_report(y_data, y_pred)

    logging.info(f'Accuracy {set_name}: {accuracy:.5f}')
    logging.info(f'Classification Report {set_name}:\n{report}')

    return accuracy
