# Final Project (TCC)

This is the Final Project for the Bachelor's degree in Computer Engineering, offered by the Federal University of Rio Grande do Norte (UFRN). It falls under the theme of Artificial Intelligence (AI) and focuses on Natural Language Processing (NLP) in the context of news classification.

Every business day, news is registered on the [UFRN Portal](https://www.ufrn.br/). Currently, there are more than 22,000 news articles available. With this in mind, in discussions with the UFRN Communication Agency (AGECOM), it was informed that the news is divided into four different categories: sciences, events, job openings, and announcements.

Thus, the purpose of this project is to present three different approaches for automatic text classification. These include three machine learning models: XGBoost, K-Means, and K-Means with dimensionality reduction.

This is the code used to train the models, and below it is explained how to replicate the models created in this project.


## Prerequisites
You need to have [Docker](https://www.docker.com/) installed. You also need accounts on [Weights & Biases](https://wandb.ai/site), [Google Drive](https://www.google.com/intl/en/drive/about.html), and [OpenAI](https://platform.openai.com/docs/overview).

On the last platform, creating an account is only necessary if you want to use the OpenAI embeddings model. The steps outlined here do not require this, as it is a time-consuming process (about 3 hours) and costly (around $1.50), and it is not necessary because the available news already has its vector representations saved in CSV format.

You need to configure two environment variables related to the created accounts. Create the `.env` file with the same content as `.env.example`, adding the values for the `WANDB_API_KEY`, found at [https://wandb.ai/authorize](https://wandb.ai/authorize), and `OPENAI_API_KEY`, found at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys), if you want to use the OpenAI API.

Additionally, you need to create model and dataset records on Weights & Biases for the code to work properly. The model records that need to be created should be named: `xgboost`, `kmeans`, and `kmeans-pca`. The datasets should be: `dataset_xgboost`, `emissions_xgboost`, `dataset_kmeans`, `emissions_kmeans`, `dataset_kmeans_pca`, and `emissions_kmeans_pca`.


## Starting the Environment
To start the environment with Docker, run the following commands in the root directory of the project.

Build the image:
```
docker build -t tcc_image .
```

Create the container:
```
docker-compose up
```

List the containers:
```
docker ps
```

In the list that appears, check if there is a container named `tcc_container`. If it exists, everything is set, and you can proceed to the next step.

Access the interactive terminal of the created container:
```
docker exec -it tcc_container /bin/bash
```

This allows you to interact with the code inside the container, and all changes made will be reflected locally, and vice versa.


## Downloading the Datasets
To version the CSV files with the data needed for model development, [DVC](https://dvc.org/) was used in conjunction with a Google Drive account.

However, based on testing, Google is blocking DVC for accounts that do not own the folder containing the file metadata. Thus, the files were placed in a Google Drive folder. To download the files to the project, run the following command:

```
gdown --folder https://drive.google.com/drive/folders/15a9iCeUTLCpUcj1a-HLFmVTQP1cx0uf- -O data
```


## Running the Code
For ease of execution, each model has a command to train it. First, navigate to the `scripts` folder of the project, then choose which models to train.

```
cd scripts
```

To train the XGBoost model:
```
python3 xgboost_train
```

To train the K-Means model:
```
python3 kmeans_train
```

To train the K-Means model with dimensionality reduction:
```
python3 kmeans_pca_train
```


## Model Interaction Environments
Three interaction environments were built, one for each proposed model. To replicate them, follow all the previous steps, then clone the following repositories and configure the necessary environment variables.

Note: For models using the K-Means algorithm, an OpenAI account with credits is required, as the models use the platform's API to generate embeddings.

- The K-Means model interaction environment is available at this [link](https://huggingface.co/spaces/AugustoOliveira099/kmeans). Environment variables to be defined: `WANDB_API_KEY` and `OPENAI_API_KEY`.

- The K-Means model interaction environment with dimensionality reduction is available at this [link](https://huggingface.co/spaces/AugustoOliveira099/kmeans-pca). Environment variables to be defined: `WANDB_API_KEY` and `OPENAI_API_KEY`.

- The XGBoost model interaction environment is available at this [link](https://huggingface.co/spaces/AugustoOliveira099/xgboost). Environment variable to be defined: `WANDB_API_KEY`.
