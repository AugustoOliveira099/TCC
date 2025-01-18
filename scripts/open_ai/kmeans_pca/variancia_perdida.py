from dotenv import load_dotenv
import joblib
import logging
import wandb
import os

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Load Weigth and Biases API key
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

wandb.login(key=WANDB_API_KEY)
run = wandb.init()
artifact = run.use_artifact('tcc-ufrn/k-means-with-pca/pca:latest', type='model')
artifact_dir = artifact.download()

# Verificar e imprimir o caminho do pca
pca_path = os.path.join(artifact_dir, 'pca.pkl')

# Verificar se o arquivo existe
if os.path.exists(pca_path):
  logging.info(f"pca file exists at: {pca_path}")
else:
  logging.error(f"pca file does NOT exist at: {pca_path}")

# Load pca
logging.info('Loading pca')
pca = joblib.load(pca_path)

# Variância explicada pelos 10 componentes selecionados
variancia_explicada = sum(pca.explained_variance_ratio_)

# Variância perdida
variancia_perdida = 1 - variancia_explicada

print(f"Variância explicada: {variancia_explicada:.2%}")
print(f"Variância perdida: {variancia_perdida:.2%}")
