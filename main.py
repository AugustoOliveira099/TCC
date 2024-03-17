import csv
import json
import requests
import logging
import time

# Endpoint das notícias
NEWS_ENDPOINT = 'https://webcache01-producao.info.ufrn.br/admin/portal-ufrn/wp-json/wp/v2/noticias-busca/?_embed'

# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def get_news() -> list:
    # Requisita as notícias e as armazena em um arquivo CSV
    try:
        # Array que armazeraná as notícias
        news = []
        # Captura as últimas 100 notícias cadastradas
        response = requests.get(f'{NEWS_ENDPOINT}&per_page=100&page=1')
        number_pages = int(response.headers['X-WP-TotalPages'])
        number_news = int(response.headers['X-WP-Total'])

        logging.info(f'There are {number_news} news.')

        # Armazena o conteúdo das notícias
        response_content = json.loads(response.content)

        for content in response_content:
            title = content['title']['rendered']
            text = content['acf']['corpo']
            news.append({ 'title': title, 'text': text, 'target': ''})

        # Captura e armazena o conteúdo das demais notícias
        logging.info('Starting news request.')
        initial_time = time.time()

        for i in range(2, number_pages + 1):
            response = requests.get(f'{NEWS_ENDPOINT}&per_page=100&page={i}')

            # Armazena o conteúdo das notícias
            response_content = json.loads(response.content)

            for content in response_content:
                title = content['title']['rendered']
                text = content['acf']['corpo']
                news.append({ 'title': title, 'text': text, 'target': ''})

        final_time = time.time()
        total_time = final_time - initial_time
        total_news = len(news)

        logging.info(f'{total_news} news items were captured in {total_time} seconds.')
        
        return news
    except requests.exceptions.ConnectionError:
        logging.error('Connection Error')
        raise
    except requests.exceptions.Timeout:
        logging.error('Timeout Error')
        raise
    except requests.exceptions.HTTPError:
        logging.error('HTTP Error')
        raise
    except Exception as e:
        logging.error('Error when requesting news: %s', str(e))
        raise

def create_csv_file(data: list) -> None:
    # Nome do arquivo CSV de saída
    file_name = 'raw_news.csv'

    # Lista das chaves do dicionário (cabeçalho do CSV)
    header = data[0].keys()

    # Escreve os dados no arquivo CSV
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        
        # Escreve o cabeçalho
        writer.writeheader()
        
        # Escreve os dados
        for line in data:
            writer.writerow(line)

    logging.info(f'CSV file generated successfully: {file_name}')


# Verifica se o script está sendo executado como programa principal
if __name__ == '__main__':
    # Requisita as notícias
    data = get_news()

    # Salva as notícias em um arquivo CSV
    create_csv_file(data)
