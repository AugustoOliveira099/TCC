import os
import sys
import logging
import pandas as pd
import wandb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from wordcloud import WordCloud
from scipy.sparse import save_npz
from xgboost import XGBClassifier
from codecarbon import track_emissions
from sklearn.preprocessing import LabelEncoder
from wandb.integration.xgboost import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.training import evaluate_model

from utils.preprocessing import lemmatize, \
                                remove_stopwords, \
                                clean_text, \
                                combine_columns

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def train_model() -> None:
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
    datafile_path = '../../data/xgboost/noticias_xgboost.csv'
    df = pd.read_csv(datafile_path)

    classified_news = pd.read_csv('../../data/classified_news.csv')
    classified_news_droped = classified_news.drop_duplicates(subset=['content'])
    classified_news_combined = combine_columns(classified_news_droped, 'title', 'content', 'combined')

    # Separa os textos (features) e os targets
    texts = df['combined']
    targets = df['target']
    texts_test = classified_news_combined['combined']
    targets_test = classified_news_combined['target']

    # Converte categorias em códigos numéricos
    label_encoder = LabelEncoder()
    targets_encoded = label_encoder.fit_transform(targets)
    targets_encoded_test = label_encoder.transform(targets_test)

    # Mostra a correspondência entre códigos numéricos e classes
    categories = []
    for code, category in enumerate(label_encoder.classes_):
        categories.append(category)
        logging.info(f'Código {code}: {category}')

    @track_emissions(save_to_api=True)
    def preprocessing_train_model():
        # Aplica o pré-processamento aos textos
        logging.info('Data preprocessing')
        processed_texts = texts.apply(clean_text).apply(remove_stopwords).apply(lemmatize)
        processed_texts_test = texts_test.apply(clean_text).apply(remove_stopwords).apply(lemmatize)

        # Converte listas de tokens de volta para string
        logging.info('Convert list of tokens to string')
        processed_texts = processed_texts.apply(lambda x: ' '.join(x))
        processed_texts_test = processed_texts_test.apply(lambda x: ' '.join(x))

        # Concatenando todos os textos em uma única string
        todos_os_textos = " ".join(processed_texts)

        # Gerando a nuvem de palavras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todos_os_textos)

        # Plotando a nuvem de palavras
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Remove os eixos
        image_path = '../../images/wordcloud_xgboost.png'
        plt.savefig(image_path)

        # Transforma os textos em vetores numéricos usando TF-IDF
        logging.info('Vectorize data')
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(processed_texts)
        X_test = vectorizer.transform(processed_texts_test)

        # Save TfidfVectorizer instance
        vectorizer_filename = 'tfidf_vectorizer.pkl'
        joblib.dump(vectorizer, vectorizer_filename)
        artifact = wandb.Artifact(name="tfidf_vectorizer", type="model")
        artifact.add_file(vectorizer_filename, name="tfidf_vectorizer.pkl")
        wandb.log_artifact(artifact)

        # Save feature matrix
        save_npz("tfidf_matrix.npz", X)
        tfidf_matrix_artifact = wandb.Artifact(name="tfidf_matrix", type="dataset")
        tfidf_matrix_artifact.add_file("tfidf_matrix.npz")
        wandb.log_artifact(tfidf_matrix_artifact)

        # # Data slipt fake
        # logging.info('Data split')
        # X_train, X_test, y_train, y_test = [X, X_test, targets_encoded, targets_encoded_test]

        # Data split
        X_train, X_teste, y_train, y_test = train_test_split(X, targets_encoded, test_size=0.1, random_state=42)

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

        # Save XGBoost model in Weights and Biases
        logging.info('Saving XGBoost model')
        xgboost_path = './xgboost.json'
        model.save_model(xgboost_path) # Save model
        logged_artifact = run.log_artifact(
            xgboost_path,
            name="xgboost",
            type="model"
        )
        run.link_artifact(
            artifact=logged_artifact,
            target_path="mlops2023-2-org/wandb-registry-model/xgboost"
        ) # Log and link the model to the Model Registry

        return model, X_train, X_teste, y_train, y_test
    
    # Train the model
    model, X_train, X_teste, y_train, y_test = preprocessing_train_model()

    # Save emssions into Weights and Biases
    logging.info('Saving carbon emissions')
    emissions_path = './emissions.csv'
    logged_artifact = run.log_artifact(
        emissions_path,
        name="emissions_xgboost",
        type="dataset"
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path="mlops2023-2-org/wandb-registry-dataset/emissions_xgboost"
    ) # Log and link the emissions to the Model Registry

    # Evaluate train model
    train_accuracy = evaluate_model(model, X_train, y_train, "train model", label_encoder)
    wandb.log({"train accuracy": train_accuracy})

    # Evaluate test model
    test_accuracy = evaluate_model(model, X_teste, y_test, "test model", label_encoder)
    wandb.log({"test accuracy": test_accuracy})

    y_pred = model.predict(X_teste)
    y_probas = model.predict_proba(X_teste)

    wandb.sklearn.plot_classifier(
        model,
        X_train,
        X_teste,
        y_train,
        y_test,
        y_pred,
        y_probas,
        label_encoder.classes_,
        model_name=f"xgboost_model",
        feature_names=None,
    )

    # Calcula a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Cria o heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Reais')
    image_path = '../../images/confusion_matrix_xgboost_teste.png'
    plt.savefig(image_path)

    # Logue o gráfico no W&B
    wandb.log({"confusion_matrix_heatmap": wandb.Image(image_path)})

    # Calculate the classification report
    report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    print(report.keys())
    del report['macro avg']
    del report['accuracy']
    del report['weighted avg']
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="Blues")
    image_path = '../../images/report_xgboost_teste.png'
    plt.savefig(image_path)

    # Logue o gráfico no W&B
    wandb.log({"Classification Report": wandb.Image(image_path)})

    # Mark the run as finished
    wandb.finish()