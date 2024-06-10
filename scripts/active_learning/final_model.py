import os
import sys
import logging
import pandas as pd
import wandb
import joblib
from wandb.integration.xgboost import WandbCallback
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from dotenv import load_dotenv


relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.training import evaluate_model

from utils.preprocessing import lemmatize, \
                                 remove_stopwords, \
                                 clean_text

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def main() -> None:
    # Login into wandb
    wandb.login(key=WANDB_API_KEY)

    # Start a run, tracking hyperparameters
    run = wandb.init(
        project="active-learning-model",
        config={
            "metric": "accuracy",
            'learning_rate': 0.2,
            'max_depth': 6,
            'reg_alpha': 5,
            'reg_lambda': 5,
            'subsample': 0.5,
            'colsample_bytree': 0.7,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'objective': 'multi:softmax',
            'num_class': 4,
        }
    )
    config = wandb.config

    logging.info('Read data')
    datafile_path = '../../data/all_classified_news.csv'
    df = pd.read_csv(datafile_path)

    # Separa os textos (features) e os targets
    texts = df['combined']
    targets = df['target']

    # Converte categorias em códigos numéricos
    label_encoder = LabelEncoder()
    targets_encoded = label_encoder.fit_transform(targets)

    # Mostra a correspondência entre códigos numéricos e classes
    for code, category in enumerate(label_encoder.classes_):
        logging.info(f'Código {code}: {category}')

    # Aplica o pré-processamento aos textos
    logging.info('Data preprocessing')
    processed_texts = texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)

    # Converte listas de tokens de volta para string
    logging.info('Convert list of tokens to string')
    processed_texts = processed_texts.apply(lambda x: ' '.join(x))

    # Transforma os textos em vetores numéricos usando TF-IDF
    logging.info('Vectorize data')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)

    # Save TfidfVectorizer instance
    vectorizer_filename = 'tfidf_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)
    artifact = wandb.Artifact(name="tfidf_vectorizer", type="model")
    artifact.add_file(vectorizer_filename, name="tfidf_vectorizer.pkl")
    wandb.log_artifact(artifact)

    logging.info('Data split')

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.1, random_state=42)

    # Criar e treinar o modelo XGBoost
    logging.info('Train the XGBoost model')
    model = XGBClassifier(
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        reg_alpha=config['reg_alpha'],
        reg_lambda=config['reg_lambda'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        eval_metric=config['eval_metric'],
        random_state=config['random_state'],
        objective=config['objective'],
        num_class=config['num_class'],
        callbacks=[WandbCallback(log_model=True)]
    )
    model.fit(X_train, y_train)

    # Evaluate train model
    train_accuracy = evaluate_model(model, X_train, y_train, "train model", label_encoder)
    wandb.log({"train accuracy": train_accuracy})

    # Evaluate test model
    test_accuracy = evaluate_model(model, X_test, y_test, "test model", label_encoder)
    wandb.log({"test accuracy": test_accuracy})

    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    wandb.sklearn.plot_classifier(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probas,
        label_encoder.classes_,
        model_name=f"active_learning_model",
        feature_names=None,
    )

    # Save model
    model.save_model('xgboost_model.json')

    # Log and link the model to the Model Registry
    run.link_model(path="./xgboost_model.json", registered_model_name="xgboost_model")

    # Mark the run as finished
    wandb.finish()

main()
