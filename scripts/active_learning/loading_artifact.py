import logging
import numpy as np
import sys
import os
import wandb
import joblib
import gradio as gr
from xgboost import XGBClassifier
from dotenv import load_dotenv

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.preprocessing import preprocess_text

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

def main() -> None:
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project="active-learning-model", job_type="inference")

    logging.info('Access and download model')
    downloaded_model_path = run.use_model('tcc-ufrn/model-registry/xgboost_model:latest')
    # print(downloaded_model_path)
    # model_dir = downloaded_model_path.download()

    # # Verificar e imprimir o caminho do modelo
    # model_path = os.path.join(model_dir, 'xgboost_model.json')

    # Verificar se o arquivo existe
    if os.path.exists(downloaded_model_path):
        logging.info(f"Model file exists at: {downloaded_model_path}")
    else:
        logging.error(f"Model file does NOT exist at: {downloaded_model_path}")
        return

    # Load model
    logging.info('Loading model')
    model = XGBClassifier()
    model.load_model(downloaded_model_path)

    logging.info('Downloading TF-IDF artifact')
    tfidf_artifact = run.use_artifact('tcc-ufrn/active-learning-model/tfidf_vectorizer:latest', type='model')
    tfidf_artifact_dir = tfidf_artifact.download()

    # Verificar e imprimir o caminho do modelo
    tfidf_path = os.path.join(tfidf_artifact_dir, 'tfidf_vectorizer.pkl')

    # Verificar se o arquivo existe
    if os.path.exists(tfidf_path):
        logging.info(f"TF-IDF file exists at: {tfidf_path}")
    else:
        logging.error(f"TF-IDF file does NOT exist at: {tfidf_path}")
        return
    
    # Load TF-IDF
    logging.info('Loading TF-IDF file')
    tfidf_vectorizer = joblib.load(tfidf_path)

    def predict(noticia):
        # Preprocess text
        noticia_clean = preprocess_text(noticia)

        # Vetorize text
        new_tfidf = tfidf_vectorizer.transform([noticia_clean])

        # Get predict
        probas = model.predict_proba(new_tfidf)
        predicted_class = np.argmax(probas)

        output = "A notícia é classificada como sendo do tema "

        if predicted_class == 0:
            output += '"Ciências" '
        elif predicted_class == 1:
            output += '"Eventos" '
        elif predicted_class == 2:
            output += '"Informes" '
        elif predicted_class == 3:
            output += '"Vagas" '
        else:
            return 'Erro: Previsão inesperada.'
        
        probability = probas[0][predicted_class]
        output += f'com uma probabilidade de {probability * 100:.2f}%.'

        return output

    description_text = "Este classificados foi feito utilizando a active learning e XGBoost.\n\n"\
                    "Submeta uma notícia para que ela seja classificada entre:\n"\
                    "- <b>Ciências:</b> Uma notícia com o tema ciências;\n"\
                    "- <b>Eventos:</b> Uma notícia que anuncia um evento;\n"\
                    "- <b>Vagas:</b> Uma notícia que anuncia vagas;\n"\
                    "- <b>Informes:</b> Uma notícia informativa;\n"\
                    "- <b>Previsão inesperada:</b> Expressa uma classificação desconhecida para a notícia.\n"

    interface = gr.Interface(fn=predict, 
                        inputs=["text"], 
                        outputs=["text"], 
                        title='Classificador de notícias', 
                        description=description_text)
        
    interface.launch()
