import pkg_resources

# Lista de dependências
dependencias = ["pandas",
                "requests",
                "tiktoken",
                "matplotlib",
                "openai",
                "dvc",
                "dvc-gdrive",
                "bs4",
                "python-dotenv",
                "numpy",
                "scikit-learn",
                "xgboost",
                "nltk",
                "spacy",
                "gradio",
                "wandb",
                "joblib",
                "google-auth",
                "google-auth-oauthlib",
                "google-auth-httplib2",
                "opentsne",
                "umap-learn",
                "codecarbon",
                "seaborn"]

# Verificando as versões instaladas
for dependencia in dependencias:
    try:
        versao = pkg_resources.get_distribution(dependencia).version
        print(f'{dependencia}: {versao}')
    except pkg_resources.DistributionNotFound:
        print(f'{dependencia}: não instalada')