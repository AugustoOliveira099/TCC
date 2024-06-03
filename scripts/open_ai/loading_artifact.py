import logging
import numpy as np
import sys
import os
import wandb
import joblib
import tiktoken
import gradio as gr
from xgboost import XGBClassifier
from dotenv import load_dotenv

relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(relative_path)

from utils.preprocessing import remove_html
from utils.data_collection import fetch_embedding

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

def main() -> None:
    # embedding model parameters
    embedding_model = "text-embedding-3-large"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-3-large
    max_tokens = 8191  # the maximum input for text-embedding-3-large is 8191

    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project="open-ai-model", job_type="inference")

    logging.info('Downloading model artifact')
    model_artifact = run.use_artifact('tcc-ufrn/open-ai-model/xgboost_model:latest', type='model')
    model_artifact_dir = model_artifact.download()

    # Verificar e imprimir o caminho do modelo
    model_path = os.path.join(model_artifact_dir, 'xgboost_model.json')

    # Verificar se o arquivo existe
    if os.path.exists(model_path):
        logging.info(f"Model file exists at: {model_path}")
    else:
        logging.error(f"Model file does NOT exist at: {model_path}")
        return

    # Load model
    logging.info('Loading model')
    model = XGBClassifier()
    model.load_model(model_path)

    logging.info('Downloading scaler artifact')
    scaler_artifact = run.use_artifact('tcc-ufrn/open-ai-model/scaler:latest', type='preprocessing')
    scaler_artifact_dir = scaler_artifact.download()

    # Verificar e imprimir o caminho do modelo
    scaler_path = os.path.join(scaler_artifact_dir, 'scaler.pkl')

    # Verificar se o arquivo existe
    if os.path.exists(scaler_path):
        logging.info(f"Scaler file exists at: {scaler_path}")
    else:
        logging.error(f"Scaler file does NOT exist at: {scaler_path}")
        return
    
    # Load scaler
    logging.info('Loading scaler')
    scaler = joblib.load(scaler_path)

    def predict(noticia):
        # Remove HTML from input
        noticia = remove_html(noticia)

        # Check the number of tokens
        encoding = tiktoken.get_encoding(embedding_encoding)
        num_tokens = len(encoding.encode(noticia))
        if num_tokens > max_tokens:
            return 'A notícia é muito longa para ser processada.'
        
        # Get embed from OpenAi API
        embed = fetch_embedding(noticia, embedding_model)
        embed = np.array(embed)

        # Normalizes data
        embed_scaled = scaler.transform(embed.reshape(1, -1))

        # Get predict
        probas = model.predict_proba(embed_scaled)
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

    description_text = "Submeta uma notícia para que ela seja classificada entre:\n"\
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

main()
