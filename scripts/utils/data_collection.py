import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# create OpenAi client
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)


def get_embedding(text: str,
                  model="text-embedding-3-large") -> list[float] | None:
    """
    Generates an embedding for the provided text using the specified model.

    Args:
        text (str): The text to be embedded. Newline characters in the text are replaced with spaces.
        model (str, optional): The model to use for generating the embedding. Defaults to "text-embedding-3-large".

    Returns:
        list[float] | None: The embedding as a list of floats if the request is successful, None if an error occurs.

    Raises:
        requests.exceptions.HTTPError: If the HTTP response indicates an error.
        requests.exceptions.ConnectionError: If there is a connection problem.
        requests.exceptions.Timeout: If the request times out.
        requests.exceptions.RequestException: For other types of request errors.
    """
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"General error occurred: {req_err}")
    return None

def fetch_embedding(text: str, embedding_model: str) -> list[float] | None:
    """
    Retrieves the embedding for the given text using the specified embedding model.

    Args:
        text (str): The text to be embedded. The function handles newline characters by replacing them with spaces.
        embedding_model (str): The model to use for generating the embedding.

    Returns:
        List[float] | None: The embedding as a list of floats if the request is successful. None if an error occurs or the embedding cannot be retrieved.

    """
    try:
        result = get_embedding(text, embedding_model)
        if result is not None:
            return result
        else:
            print(f"Failed to retrieve data from the following news:\n {text}")
    except Exception as e:
        print('An error ocurred while processing the news.')
