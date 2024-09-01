# Trabalho de Conclusão de Curso (TCC)
Este é o Trabalho de Conlusão de Curso para o curso Bacharelado em Engenharia de Computação, oferecido pela Universidade Federal do Rio Grande do Norte (UFRN). Ele está inserido no tema de Inteligência Artificial (IA) e trata a respeito do Processamento de Linguagem Natural (PLN) no contexto de classificação de notícias. 

Todos os dias úteis são cadastradas notícias no [Portal da UFRN](https://www.ufrn.br/). Atualmente, existem mais de 22 mil notícias disponíveis. Tendo-se isso em vista, em conversa com a Agência de Comunicação da UFRN (AGECOM), foi informado que as notícias se dividem em quatro temas diferentes: ciências, eventos, vagas e informes.

Desse modo, o intuito deste trabalho é apresentar três abordagens diferentes para a classificação automática dos textos. São três modelos de aprendizado de máquina: XGBoost, K-Means e K-Means com a adição da redução de dimensionalidade.

Este é o código utilizado para o treinamento dos modelos, e a seguir estará exposto como é possível replicar os modelos criados no TCC em questão.


## Pré-requisitos
É necessário ter o [Docker](https://www.docker.com/) instalado. Contas no [Weights & Biases](https://wandb.ai/site), [Google Drive](https://www.google.com/intl/pt-br/drive/about.html) e [OpenAI](https://platform.openai.com/docs/overview). 

Nesta última plataforma, é apenas necessário criar uma conta caso queira utilizar o modelo de embeddings da OpenAI. Os passos aqui elucidados não utilizam, uma vez que é um processo custoso em tempo (cerca de 3 horas) e em dinheiro (em torno de U$ 1,50), além de não ser necessário, pois as notícias disponíveis já possuem suas representações vetoriais salvas em formato CSV. 

É necessário configurar duas variáveis de ambiente relacionadas às contas criadas. Crie o arquivo ``.env`` com o mesmo conteúdo presente em ``.env.example``, adicionando os valores para as variáveis ``WANDB_API_KEY``, encontrada em [https://wandb.ai/authorize](https://wandb.ai/authorize), e ``OPENAI_API_KEY``, presente em [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys), caso queira utilizar a API da OpenAI.

Ademais, é necessário criar registros de modelos e conjunto de dados no Weights & Biases para que o código funcione adequadamente. Os registros de modelos que precisam ser criados devem ter os seguintes nomes: ``xgboost``, ``kmeans`` e ``kmeans-pca``. Já os conjuntos de dados devem ser: ``dataset_xgboost``, ``emissions_xgboost``, ``dataset_kmeans``, ``emissions_kmeans``, ``dataset_kmeans_pca`` e ``emissions_kmeans_pca``.


## Iniciando o ambiente
Para iniciar o ambiente com Docker, basta executar os comandos a seguir na raiz do projeto.

Contrói a imagem:
```
docker build -t tcc_image .
```

Cria o conatainer:
```
docker-compose up
```

Exibe a lista de de containers:
```
docker ps
```

Na lista que irá aparecer, é necessário observar se existe container com o nome de ``tcc_container``, caso exista, está tudo certo e é possível ir para o passo seguinte.

Acessa o terminal interativo do container criado:
```
docker exec -it tcc_container /bin/bash
```

Com isso, é possível interagir com o código presente no container e todas as mudanças feitas nele serão refletidas localmente, e vice-versa.


## Download dos conjuntos de dados
Para versionar os arquivos CSV com os dados necessários para o desenvolvimento dos modelos, foi utilizado o [DVC](https://dvc.org/) em conjunto com uma conta do Google Drive. Para fazer o download dos dados necessários para treinar os modelos, basta executar o comando a seguir:

```
dvc pull
```

Será solicitado login com sua conta do Google para ter acesso aos arquivos presentes no Google Drive. Após isso, o download é iniciado de maneira automática.


## Executando o código
Visando a facilidade da execução, cada modelo possui um comando para ser treinado. Primeiro é necessário acessar a pasta de ``script`` do projeto, para que em seguida seja possível escolher quais modelos treinar.

```
cd scripts
```

Para treinar o modelo XGBoost:
```
python3 xgboost_train
```

Para treinar o modelo K-Means:
```
python3 kmeans_train
```

Para treinar o modelo K-Means com redução de dimensionalidade:
```
python3 kmeans_pca_train
```




- Inicia um projeto dvc
```
dvc init
```

- Adiciona o rastreamento
```
dvc add data/second_part.csv
```

- Define o armazenamento --default (-d) para os dados
```
dvc remote add -d storing_data gdrive://1LiXbvMkEhfMvHgGBfBcybAO1XoBMycwQ
```

- Salva os dados no google drive
```
dvc pull
```

