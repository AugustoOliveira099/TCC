import os
import logging
import subprocess
import pandas as pd

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {e}")
        exit(1)

def configure_dvc_remote():
    try:
        # Adicionar remoto DVC, caso ainda não exista
        logging.info(f'dvc remote add -d myremote gdrive://{GDRIVE_FOLDER_ID}')
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", f"gdrive://{GDRIVE_FOLDER_ID}"], check=True)
    except subprocess.CalledProcessError:
        # Se o remoto já existir, ignore o erro
        pass
    try:
        # Configurar o remoto DVC para usar a conta de serviço
        logging.info('dvc remote modify myremote gdrive_use_service_account true')
        subprocess.run(["dvc", "remote", "modify", "myremote", "gdrive_use_service_account", "true"], check=True)

        logging.info(f'dvc remote modify myremote --local gdrive_service_account_json_file_path {SERVICE_ACCOUNT_JSON}')
        subprocess.run(["dvc", "remote", "modify", "myremote", "--local", "gdrive_service_account_json_file_path", SERVICE_ACCOUNT_JSON], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao configurar o DVC remoto: {e}")
        pass
        # raise